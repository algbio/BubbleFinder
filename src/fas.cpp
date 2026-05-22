#include "fas.h"

void FeedbackArcSet::run_fas(const spqr_compat::Graph &graph, 
                             std::vector<spqr_compat::edge> &result)
{
    spqr_compat::NodeArray<int>  disc(graph, 0), dfsNum(graph, 0);
    spqr_compat::NodeArray<char> colour(graph, 0);
    spqr_compat::EdgeArray<EdgeType> etype(graph);
    int discTimer = 0, dfsIndex = 0;
    std::vector<spqr_compat::node> dfsNumInverse(graph.numberOfNodes(), nullptr);
    spqr_compat::edge singleBackEdge = nullptr;

    std::function<void(spqr_compat::node)> dfs1 = [&](spqr_compat::node u)
    {
        colour[u] = 1;
        disc[u]  = ++discTimer;
        if (dfsNum[u] == 0) {
            dfsNum[u] = dfsIndex;
            dfsNumInverse[dfsIndex] = u;
            ++dfsIndex;
        }
        
        graph.forEachAdj(u, [&](node v, edge e) {
            if (graph.source(e) != u) return;  
            
            if (colour[v] == 0) {
                etype[e] = TREE;
                dfs1(v);
            } else if (colour[v] == 1) {
                etype[e] = BACK;
            } else {
                etype[e] = (disc[u] < disc[v]) ? FORWARD : CROSS;
            }
        });
        colour[u] = 2;
    };
    
    spqr_compat::node root = graph.firstNode();
    dfs1(root);
    
    spqr_compat::Graph T;
    spqr_compat::NodeArray<spqr_compat::node> mapToTree(graph, nullptr);
    spqr_compat::NodeArray<spqr_compat::node> mapToOriginal(T, nullptr);
    for (spqr_compat::node v : graph.nodes) {
        auto newNode = T.newNode();
        mapToTree[v] = newNode;
        mapToOriginal[newNode] = v;
    }

    spqr_compat::NodeArray<int> backedgeTailSubtreeCount(T, 0);
    int backEdgeCount = 0;

    spqr_compat::node y = nullptr;

    for (spqr_compat::edge e : graph.edges) {
        spqr_compat::node u = graph.source(e);
        spqr_compat::node v = graph.target(e);
        
        if (etype[e] == TREE) {
            T.newEdge(mapToTree[u], mapToTree[v]);
        } else if (etype[e] == BACK) {
            singleBackEdge = e;
            ++backEdgeCount;
            backedgeTailSubtreeCount[mapToTree[u]]++;
            if (y == nullptr || dfsNum[v] > dfsNum[y]) {
                y = v;
            }
        }
    }

    if (backEdgeCount == 1) {
        result.push_back(singleBackEdge);
    }

    spqr_compat::node z = graph.firstNode(); 
    {
        std::function<void(spqr_compat::node, spqr_compat::node)> dfsSubtree =
            [&](spqr_compat::node u, spqr_compat::node prev) {
                T.forEachAdj(u, [&](node v, edge e) {
                    if (T.source(e) != u) return;
                    if (v == prev) return;
                    dfsSubtree(v, u);
                    backedgeTailSubtreeCount[u] += backedgeTailSubtreeCount[v];
                });
            };

        dfsSubtree(mapToTree[root], nullptr);

        std::function<void(spqr_compat::node, spqr_compat::node)> dfs2 =
            [&](spqr_compat::node u, spqr_compat::node prev) {
                int c = backedgeTailSubtreeCount[u];
                if (c == backEdgeCount) {
                    z = mapToOriginal[u];
                }
                T.forEachAdj(u, [&](node v, edge e) {
                    if (T.source(e) != u) return;
                    if (v == prev) return;
                    dfs2(v, u);
                });
            };

        dfs2(mapToTree[root], nullptr);

        assert(z != nullptr);
    }

    if (dfsNum[y] > dfsNum[z]) {
        return;
    }

    {
        std::function<bool(spqr_compat::node)> inInterval = [&](spqr_compat::node u) -> bool {
            return dfsNum[y] <= dfsNum[u] && dfsNum[u] <= dfsNum[z];
        };

        spqr_compat::Graph Gp;
        spqr_compat::NodeArray<spqr_compat::node> mapToGp(graph, nullptr);
        for (spqr_compat::node v : graph.nodes)
            mapToGp[v] = Gp.newNode();

        for (spqr_compat::edge e : graph.edges) {
            spqr_compat::node u = graph.source(e);
            spqr_compat::node v = graph.target(e);
            if (inInterval(u) && inInterval(v) && etype[e] == EdgeType::TREE)
                continue;
            Gp.newEdge(mapToGp[u], mapToGp[v]);
        }

        if (!isAcyclic(Gp)) {
            return;
        }
    }

    spqr_compat::NodeArray<bool> loop(graph, false);
    spqr_compat::NodeArray<int> maxi(graph, 0);
    {
        spqr_compat::NodeArray<int> discDfs(graph), fin(graph);
        int timeCounter = 0;

        std::function<bool(spqr_compat::node, spqr_compat::node)> isDescendant =
            [&](spqr_compat::node ancestor, spqr_compat::node v) -> bool {
                return ancestor != v &&
                       discDfs[ancestor] < discDfs[v] &&
                       fin[v] < fin[ancestor];
            };

        spqr_compat::NodeArray<bool> vis(graph, false);
        std::function<void(spqr_compat::node)> dfsAssignTime =
            [&](spqr_compat::node u) {
                vis[u] = true;
                discDfs[u] = ++timeCounter;
                graph.forEachAdj(u, [&](node v, edge e) {
                    if (graph.source(e) != u) return;
                    if (vis[v]) return;
                    dfsAssignTime(v);
                });
                fin[u] = ++timeCounter;
            };
        
        dfsAssignTime(root);

        vis = spqr_compat::NodeArray<bool>(graph, false);

        std::vector<spqr_compat::node> stk;
        stk.reserve(graph.numberOfNodes());
        std::function<void(spqr_compat::node)> dfsOrder =
            [&](spqr_compat::node u) {
                vis[u] = true;
                graph.forEachAdj(u, [&](node v, edge e) {
                    if (graph.source(e) != u) return;
                    if (!vis[v]) dfsOrder(v);
                });
                stk.push_back(u);
            };

        dfsOrder(root);

        std::reverse(stk.begin(), stk.end());
        while (!stk.empty()) {
            spqr_compat::node u = stk.back();
            stk.pop_back();
            
            graph.forEachAdj(u, [&](node v, edge e) {
                if (graph.source(e) != u) return;

                if (etype[e] == EdgeType::BACK) {
                    return;
                }

                if (dfsNum[y] <= dfsNum[u] &&
                    dfsNum[u]   < dfsNum[z] &&
                    dfsNum[u] + 1 == dfsNum[v]) {
                    return;
                }

                if (dfsNum[v] <= dfsNum[z]) {
                    maxi[u] = std::max(maxi[u], dfsNum[v]);
                } else {
                    maxi[u] = std::max(maxi[u], maxi[v]);
                }
            });

            if (isDescendant(z, u)) {
                loop[u] = true;
            } else {
                graph.forEachAdj(u, [&](node v, edge e) {
                    if (graph.source(e) != u) return;
                    if (etype[e] != EdgeType::BACK) {
                        loop[u] = loop[u] || (dfsNum[v] > dfsNum[z] && loop[v]);
                    }
                });
            }
        }
    }

    spqr_compat::node v = y;
    int maxitest = -1;
    for (int i = 0; i < dfsNum[y]; ++i) {
        maxitest = std::max(maxitest, maxi[dfsNumInverse[i]]);
    }

    while (true) {
        if (loop[v]) {
            break;
        }

        maxitest = std::max(maxitest, maxi[v]);

        if (maxitest <= dfsNum[v]) {
            int        cntVnext   = 0;
            spqr_compat::edge edgeToAdd  = nullptr;

            graph.forEachAdj(v, [&](node v2, edge e) {
                if (graph.source(e) != v) return;
                
                if (dfsNum[v] + 1 == dfsNum[v2]) {
                    if (edgeToAdd != nullptr) {
                        edgeToAdd = nullptr;
                        return; 
                    } 
                    edgeToAdd = e;
                    ++cntVnext;
                }
            });
            if (cntVnext == 1 && edgeToAdd != nullptr) {
                result.push_back(edgeToAdd);
            }
        }

        if (dfsNum[v] < dfsNum[z] - 1) {
            v = dfsNumInverse[dfsNum[v] + 1];
        } else {
            break;
        }
    }

    return;
}


void FeedbackArcSet::find_feedback_arcs(
    std::vector<spqr_compat::edge> &result,
    const spqr_compat::NodeArray<bool> &toRemove
) {
    spqr_compat::Graph newG;
    spqr_compat::NodeArray<spqr_compat::node> orig2copy(G, nullptr);
    spqr_compat::EdgeArray<spqr_compat::edge> copy2origE(newG, nullptr);

    for (spqr_compat::node v : G.nodes) {
        if (!toRemove[v]) {
            orig2copy[v] = newG.newNode();
        }
    }

    for (spqr_compat::edge e : G.edges) {
        spqr_compat::node u  = G.source(e);
        spqr_compat::node w  = G.target(e);
        spqr_compat::node uC = orig2copy[u];
        spqr_compat::node wC = orig2copy[w];
        if (uC && wC) {
            spqr_compat::edge f = newG.newEdge(uC, wC);
            copy2origE[f] = e;
        }
    }

    std::vector<spqr_compat::edge> copyResult;
    run_fas(newG, copyResult);

    result.clear();
    result.reserve(copyResult.size());
    for (spqr_compat::edge f : copyResult) {
        result.push_back(copy2origE[f]);
    }
}


std::vector<spqr_compat::edge> FeedbackArcSet::run() {
    spqr_compat::NodeArray<int> comp(this->G);
    int sccs = strongComponents(this->G, comp);

    std::vector<int> size(sccs, 0);
    for (spqr_compat::node v : this->G.nodes) ++size[comp[v]];

    int nonTrivial = 0;
    int ntIdx      = -1;

    for (int i = 0; i < sccs; ++i) {
        if (size[i] > 1) {
            ++nonTrivial;
            ntIdx = i;
        }
    }

    if (nonTrivial >= 2) {
        return {};
    } else if (nonTrivial == 1) {
        spqr_compat::NodeArray<bool> toRemove(this->G, false);
        for (spqr_compat::node v : this->G.nodes) {
            if (comp[v] != ntIdx) toRemove[v] = true;
        }

        std::vector<spqr_compat::edge> res;
        this->find_feedback_arcs(res, toRemove);
        return res;
    } else {
        std::vector<spqr_compat::edge> res;
        for (spqr_compat::edge e : this->G.edges) res.push_back(e);
        return res;
    }
}

bool FeedbackArcSet::run_or_acyclic(std::vector<spqr_compat::edge> &out) {
    spqr_compat::NodeArray<int> comp(this->G);
    int sccs = strongComponents(this->G, comp);

    std::vector<int> size(sccs, 0);
    for (spqr_compat::node v : this->G.nodes) ++size[comp[v]];

    int nonTrivial = 0;
    int ntIdx      = -1;

    for (int i = 0; i < sccs; ++i) {
        if (size[i] > 1) {
            ++nonTrivial;
            ntIdx = i;
        }
    }

    if (nonTrivial >= 2) {
        out.clear();
        return false;  
    } else if (nonTrivial == 1) {
        spqr_compat::NodeArray<bool> toRemove(this->G, false);
        for (spqr_compat::node v : this->G.nodes) {
            if (comp[v] != ntIdx) toRemove[v] = true;
        }

        out.clear();
        this->find_feedback_arcs(out, toRemove);
        return false;  
    } else {
        out.clear();
        return true; 
    }
}
