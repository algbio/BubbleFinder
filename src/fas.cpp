#include "fas.h"

void FeedbackArcSet::run_fas(const ogdf::Graph &graph, 
                             std::vector<ogdf::edge> &result)
{
    ogdf::NodeArray<int>  disc(graph, 0), dfsNum(graph, 0);
    ogdf::NodeArray<char> colour(graph, 0);
    ogdf::EdgeArray<EdgeType> etype(graph);
    int discTimer = 0, dfsIndex = 0;
    std::vector<ogdf::node> dfsNumInverse(graph.numberOfNodes(), nullptr);
    ogdf::edge singleBackEdge = nullptr;

    // Building dfs tree and classifying edges
    std::function<void(ogdf::node)> dfs1 = [&](ogdf::node u)
    {
        colour[u] = 1;
        disc[u]  = ++discTimer;
        if (dfsNum[u] == 0) {
            dfsNum[u] = dfsIndex;
            dfsNumInverse[dfsIndex] = u;
            ++dfsIndex;
        }
        
        for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
        {
            if (!adj->isSource()) continue;
            ogdf::edge e = adj->theEdge();
            ogdf::node v = adj->twinNode();
            
            if (colour[v] == 0) {
                etype[e] = TREE;
                dfs1(v);
            } else if (colour[v] == 1) {
                etype[e] = BACK;
            } else {
                etype[e] = (disc[u] < disc[v]) ? FORWARD : CROSS;
            }
        }
        colour[u] = 2;
    };
    
    ogdf::node root = graph.firstNode();
    dfs1(root);
    
    ogdf::Graph T;
    ogdf::NodeArray<ogdf::node> mapToTree(graph, nullptr);
    ogdf::NodeArray<ogdf::node> mapToOriginal(T, nullptr);
    for (ogdf::node v : graph.nodes) {
        auto newNode = T.newNode();
        mapToTree[v] = newNode;
        mapToOriginal[newNode] = v;
    }

    ogdf::NodeArray<int> backedgeTailSubtreeCount(T, 0);
    int backEdgeCount = 0;

    ogdf::node y = nullptr;

    for (ogdf::edge e : graph.edges) {
        ogdf::node u = e->source();
        ogdf::node v = e->target();
        
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

    ogdf::node z = graph.firstNode(); 
    {
        std::function<void(ogdf::node, ogdf::node)> dfsSubtree =
            [&](ogdf::node u, ogdf::node prev) {
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;
                    ogdf::node v = adj->twinNode();
                    if (v == prev) continue;
                    dfsSubtree(v, u);
                    backedgeTailSubtreeCount[u] += backedgeTailSubtreeCount[v];
                }
            };

        dfsSubtree(mapToTree[root], nullptr);

        std::function<void(ogdf::node, ogdf::node)> dfs2 =
            [&](ogdf::node u, ogdf::node prev) {
                int c = backedgeTailSubtreeCount[u];
                if (c == backEdgeCount) {
                    z = mapToOriginal[u];
                }
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;
                    ogdf::node v = adj->twinNode();
                    if (v == prev) continue;
                    dfs2(v, u);
                }
            };

        dfs2(mapToTree[root], nullptr);

        assert(z != nullptr);
    }

    if (dfsNum[y] > dfsNum[z]) {
        return;
    }

    {
        std::function<bool(ogdf::node)> inInterval = [&](ogdf::node u) -> bool {
            return dfsNum[y] <= dfsNum[u] && dfsNum[u] <= dfsNum[z];
        };

        ogdf::Graph Gp;
        ogdf::NodeArray<ogdf::node> mapToGp(graph, nullptr);
        for (ogdf::node v : graph.nodes)
            mapToGp[v] = Gp.newNode();

        for (ogdf::edge e : graph.edges) {
            ogdf::node u = e->source();
            ogdf::node v = e->target();
            if (inInterval(u) && inInterval(v) && etype[e] == EdgeType::TREE)
                continue;
            Gp.newEdge(mapToGp[u], mapToGp[v]);
        }

        if (!isAcyclic(Gp)) {
            // Graph without interval is not acyclic, so there is no feedback set
            return;
        }
    }

    ogdf::NodeArray<bool> loop(graph, false);
    ogdf::NodeArray<int> maxi(graph, 0);
    {
        ogdf::NodeArray<int> discDfs(graph), fin(graph);
        int timeCounter = 0;

        std::function<bool(ogdf::node, ogdf::node)> isDescendant =
            [&](ogdf::node ancestor, ogdf::node v) -> bool {
                return ancestor != v &&
                       discDfs[ancestor] < discDfs[v] &&
                       fin[v] < fin[ancestor];
            };

        ogdf::NodeArray<bool> vis(graph, false);
        std::function<void(ogdf::node)> dfsAssignTime =
            [&](ogdf::node u) {
                vis[u] = true;
                discDfs[u] = ++timeCounter;
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue; // follow children only
                    ogdf::node v = adj->twinNode();
                    if (vis[v]) continue;
                    dfsAssignTime(v);
                }
                fin[u] = ++timeCounter;
            };
        
        dfsAssignTime(root);

        vis = ogdf::NodeArray<bool>(graph, false);

        std::vector<ogdf::node> stk;
        stk.reserve(graph.nodes.size());
        std::function<void(ogdf::node)> dfsOrder =
            [&](ogdf::node u) {
                vis[u] = true;
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;
                    ogdf::node v = adj->twinNode();
                    if (!vis[v]) dfsOrder(v);
                }
                stk.push_back(u);
            };

        dfsOrder(root);

        std::reverse(stk.begin(), stk.end());
        while (!stk.empty()) {
            ogdf::node u = stk.back();
            stk.pop_back();
            
            // maxi computation
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                if (!adj->isSource()) continue;

                ogdf::edge e  = adj->theEdge();
                ogdf::node v  = adj->twinNode();

                if (etype[e] == EdgeType::BACK) {
                    continue;
                }

                if (dfsNum[y] <= dfsNum[u] &&
                    dfsNum[u]   < dfsNum[z] &&
                    dfsNum[u] + 1 == dfsNum[v]) {
                    continue;
                }

                if (dfsNum[v] <= dfsNum[z]) {
                    maxi[u] = std::max(maxi[u], dfsNum[v]);
                } else { // dfsNum[v] > dfsNum[z]
                    maxi[u] = std::max(maxi[u], maxi[v]);
                }
            }

            // loop computation
            if (isDescendant(z, u)) {
                loop[u] = true;
            } else {
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;

                    ogdf::edge e  = adj->theEdge();
                    ogdf::node v  = adj->twinNode();

                    if (etype[e] != EdgeType::BACK) {
                        loop[u] |= (dfsNum[v] > dfsNum[z] && loop[v]);
                    }
                }
            }
        }
    }

    // init
    ogdf::node v = y;
    int maxitest = -1;
    for (int i = 0; i < dfsNum[y]; ++i) {
        maxitest = std::max(maxitest, maxi[dfsNumInverse[i]]);
    }

    // step 4
    while (true) {
        if (loop[v]) {
            break;
        }

        maxitest = std::max(maxitest, maxi[v]);

        if (maxitest <= dfsNum[v]) {
            int        cntVnext   = 0;
            ogdf::edge edgeToAdd  = nullptr;

            for (ogdf::adjEntry adj = v->firstAdj(); adj; adj = adj->succ()) {
                if (!adj->isSource()) continue;
                
                ogdf::edge e  = adj->theEdge();
                ogdf::node v2 = adj->twinNode();
                
                if (dfsNum[v] + 1 == dfsNum[v2]) {
                    if (edgeToAdd != nullptr) {
                        edgeToAdd = nullptr;
                        break;
                    } 
                    edgeToAdd = e;
                    ++cntVnext;
                }
            }
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
    std::vector<ogdf::edge> &result,
    const ogdf::NodeArray<bool> &toRemove
) {
    ogdf::Graph newG;
    ogdf::NodeArray<ogdf::node> orig2copy(G, nullptr);
    ogdf::EdgeArray<ogdf::edge> copy2origE(newG, nullptr);

    for (ogdf::node v : G.nodes) {
        if (!toRemove[v]) {
            orig2copy[v] = newG.newNode();
        }
    }

    for (ogdf::edge e : G.edges) {
        ogdf::node u  = e->source();
        ogdf::node w  = e->target();
        ogdf::node uC = orig2copy[u];
        ogdf::node wC = orig2copy[w];
        if (uC && wC) {
            ogdf::edge f = newG.newEdge(uC, wC);
            copy2origE[f] = e;
        }
    }

    std::vector<ogdf::edge> copyResult;
    run_fas(newG, copyResult);

    result.clear();
    result.reserve(copyResult.size());
    for (ogdf::edge f : copyResult) {
        result.push_back(copy2origE[f]);
    }
}


std::vector<ogdf::edge> FeedbackArcSet::run() {
    ogdf::NodeArray<int> comp(this->G);
    int sccs = strongComponents(this->G, comp);

    std::vector<int> size(sccs, 0);
    for (ogdf::node v : this->G.nodes) ++size[comp[v]];

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
        ogdf::NodeArray<bool> toRemove(this->G, false);
        for (ogdf::node v : this->G.nodes) {
            if (comp[v] != ntIdx) toRemove[v] = true;
        }

        std::vector<ogdf::edge> res;
        this->find_feedback_arcs(res, toRemove);
        return res;
    } else {
        std::vector<ogdf::edge> res;
        for (ogdf::edge &e : this->G.edges) res.push_back(e);
        return res;
    }
}