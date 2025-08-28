#include "fas.h"

void FeedbackArcSet::run_fas(ogdf::Graph G, 
    std::vector<ogdf::edge> &result
) {
    ogdf::NodeArray<int>  disc(G, 0), dfsNum(G, 0);
    ogdf::NodeArray<ogdf::node> parent(G, nullptr);
    ogdf::NodeArray<char> colour(G, 0);
    ogdf::EdgeArray<EdgeType> etype(G);   
    int time = 0, dfsTime=0;
    std::vector<ogdf::node> dfsNumInverse(G.maxNodeIndex()+1, nullptr);
    ogdf::edge singleBackEdge = nullptr;

    
    // Building dfs tree and classifying edges
    std::function<void(ogdf::node)> dfs1 = [&](ogdf::node u)
    {
        colour[u] = 1;
        disc[u]  = ++time;
        if(dfsNum[u] == 0) {
            dfsNum[u] = dfsTime;
            dfsNumInverse[dfsTime] = u;
            dfsTime++;
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
        // finish[u] = ++time;
    };
    
    ogdf::node root = G.firstNode();
    dfs1(root);
    
    
    
    ogdf::Graph T;
    ogdf::NodeArray<ogdf::node> map(G, nullptr);
    ogdf::NodeArray<ogdf::node> mapInverse(T, nullptr);
    for (ogdf::node v : G.nodes) {
        auto newNode = T.newNode();
        map[v] = newNode;
        mapInverse[newNode] = v;
    }


    ogdf::NodeArray<int> backedgeTailSubtreeCount(G, 0);
    int backEdgeCount = 0;



    ogdf::node y = nullptr;

    for(auto &e:G.edges) {
        ogdf::node u = e->source();
        ogdf::node v = e->target();
        
        if(etype[e] == TREE) {
            T.newEdge(map[u], map[v]);
            // std::cout << "Adding tree edge: " << map[u]->index() << " -> " << map[v]->index() << std::endl;
        } else if(etype[e] == BACK) {
            singleBackEdge = e;
            backEdgeCount++;
            backedgeTailSubtreeCount[map[u]]++;
            if(y == nullptr || dfsNum[v]>dfsNum[y]) {
                y=v;
            }
        }
    }


    if(backEdgeCount == 1) {
        result.push_back(singleBackEdge);
    }


    // std::cout << "Total back edges: " << backEdgeCount << std::endl;
    // for(auto &u:G.nodes) {
    //     std::cout << "Node " << u->index() << ": back edges = " << backedgeTailSubtreeCount[u] << std::endl;
    // }
    // return;
    // 


    // drawGraph(G, "input");
    // drawGraph(T, "treeWithTreeEdges");
    ogdf::node z = G.firstNode(); 
    {

        std::function<void(ogdf::node, ogdf::node)> dfsSubtree = [&](ogdf::node u, ogdf::node prev=nullptr) {
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                if (!adj->isSource()) continue;
                ogdf::node v = adj->twinNode();
                if (v == prev) continue;
                dfsSubtree(v, u);
                backedgeTailSubtreeCount[u] += backedgeTailSubtreeCount[v];  // accumulate back edges from child
            }
        };

        dfsSubtree(map[root], nullptr);

        // for(auto &u:G.nodes) {
        //     std::cout << "Node " << u->index() << ": back edges in subtree = " << backedgeTailSubtreeCount[u] << std::endl;
        // }   

        std::function<void(ogdf::node, ogdf::node)> dfs2 = [&](ogdf::node u, ogdf::node prev=nullptr) -> void {
            int c = backedgeTailSubtreeCount[u];
            if (c == backEdgeCount) {
                z = mapInverse[u];
                // std::cout << mapInverse[u]->index() << " is a candidate for deepest node in back edges." << std::endl;
            }
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
                if (adj->isSource()) {
                    ogdf::node v = adj->twinNode();
                    if(v == prev) continue;
                    dfs2(v, u);
                }
            // std::cout << "Visiting node " << u->index() << ": back edges in subtree = " << c << std::endl;
            // return c;
        };


        dfs2(map[root], nullptr);

        // std:: cout << "Done" << std::endl;

        assert(z != nullptr);
        
        // std::cout << "Deepest node in back edges: " << z->index() << " with " << backedgeTailSubtreeCount[z] << " back edges in its subtree." << std::endl;
        


        // std::cout << "Deepest node in back edges: " << z << " with " << backedgeTailSubtreeCount[z] << " back edges in its subtree." << std::endl;
    }

    // std::cout << "y:" << y << " z:" << z << std::endl;

    // std::cout << G.firstNode() << std::endl;
    // std::cout << y << ", " << z << std::endl;
    if(dfsNum[y]>dfsNum[z]) {
        // std::cout << "y > z, no feedback vertices" << std::endl;
        return;
    }

    // removing [y,z] and checking acyclicity
    {
        std::function<bool(ogdf::node)> inInterval = [&](ogdf::node u) -> bool {
            return dfsNum[y] <= dfsNum[u] && dfsNum[u] <= dfsNum[z];
        };
        ogdf::Graph Gp;
        ogdf::NodeArray<ogdf::node> map(G, nullptr);
        for (ogdf::node v : G.nodes)
            map[v] = Gp.newNode();


        for (ogdf::edge e : G.edges) {
            ogdf::node u = e->source(), v = e->target();
            // std::cout << u << " -> " << v << std::endl;
            // std::cout << (inInterval(u)) << " " << (inInterval(v)) << std::endl; 
            if(inInterval(u) && inInterval(v) && etype[e]==EdgeType::TREE) continue;
            // std::cout << "added" << std::endl;
            // if(!inInterval(u) && !inInterval(v)) 
            Gp.newEdge(map[u], map[v]);
        }

        GraphIO::drawGraph(Gp, "removed_input"+std::to_string(G.nodes.size()));

        if(!isAcyclic(Gp)) {
            // std::cout << "Graph without interval is not acylic, so there is no feedback set" << std::endl;
            return;
        }
    }


    ogdf::NodeArray<bool> loop(G, false);
    ogdf::NodeArray<int> maxi(G, 0);
    {
        ogdf::NodeArray<int> disc(G), fin(G);
        int time=0;
        std::function<bool(ogdf::node, ogdf::node)> isDescendant = [&](ogdf::node ancestor, ogdf::node v) -> bool {
            return ancestor != v && disc[ancestor] < disc[v] && fin[v] < fin[ancestor];
        };


        ogdf::NodeArray<bool> vis(G, false);
        std::function<void(ogdf::node)> dfsTime = [&](ogdf::node u) {
            vis[u] = true;
            disc[u] = ++time;
            // std::cout << "at " << u << std::endl;
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
                if (adj->isSource()) {            // follow children only
                    ogdf::node v = adj->twinNode();
                    if (vis[v]) continue;
                    dfsTime(v);
                }
            fin[u] = ++time;
            
        };
        
        dfsTime(root);
        // return ;

        vis = ogdf::NodeArray<bool>(G, false);

        std::vector<ogdf::node> stk;
        stk.reserve(G.nodes.size());
        std::function<void(ogdf::node)> dfsOrder = [&](ogdf::node u) {
            vis[u] = true;
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                if (adj->isSource()) {
                    ogdf::edge e = adj->theEdge();
                    ogdf::node v = adj->twinNode();
                    if (!vis[v]) dfsOrder(v);
                }
            }
            stk.push_back(u);
        };

        dfsOrder(root);
        // std::cout << stk.size() << " size "<< std::endl;

        std::reverse(stk.begin(), stk.end());
        while(!stk.empty()) {
            ogdf::node u = stk.back();
            stk.pop_back();
            // std::cout << "at " << u << std::endl;

            
            // maxi computation
            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                if (!adj->isSource()) continue;

                ogdf::edge e = adj->theEdge();
                ogdf::node v  = adj->twinNode();

                // std::cout << dfsNum[u] << " -> " << dfsNum[v] << ": ";
                if(etype[e]==EdgeType::BACK) {
                    // std::cout << "backedge" << std::endl;
                    continue;
                }
                // if(dfsNum[y]<=dfsNum[v] && 
                //     dfsNum[v]<=dfsNum[z] && 
                //     dfsNum[y]<=dfsNum[u] && 
                //     dfsNum[u]<dfsNum[z] && 
                //     etype[e] == EdgeType::TREE) {
                //     std::cout << "tree edge, ignore" << std::endl;
                //     continue;
                // }

                if(dfsNum[y]<=dfsNum[u] && 
                    dfsNum[u]<dfsNum[z] &&
                    dfsNum[u]+1 == dfsNum[v]) {
                        continue;
                    }

                if(dfsNum[v]<=dfsNum[z]) {
                    maxi[u] = std::max(maxi[u], dfsNum[v]);
                    // std::cout << dfsNum[v] << std::endl;
                } else if(dfsNum[v]>dfsNum[z]) {
                    maxi[u] = std::max(maxi[u], maxi[v]);
                    // std::cout << maxi[v] << std::endl;
                }

            }


            // loop computation
            if(isDescendant(z, u)) {
                loop[u] = true;
                // std::cout << " it is proper desc of z" << std::endl; 
            } else {
                for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;

                    ogdf::edge e = adj->theEdge();
                    ogdf::node v  = adj->twinNode();

                    if(etype[e]!=EdgeType::BACK) {
                        loop[u] |= (dfsNum[v]>dfsNum[z] && loop[v]);
                    }
                }
            }
        }
    }

    
    
    // init
    ogdf::node v=y;
    int maxitest=-1;
    for (int i = 0; i < dfsNum[y]; i++) {
        maxitest=std::max(maxitest, maxi[dfsNumInverse[i]]);
    }

    // assert(maxitest >= dfsNum[y]);

    // step 4
    while(true) {
        if(loop[v] == true) {
            break;
        }

        maxitest = std::max(maxitest, maxi[v]);


        if(maxitest<=dfsNum[v]) {
            int cntVnext=0;
            ogdf::edge edgeToAdd=nullptr;
            for (ogdf::adjEntry adj = v->firstAdj(); adj; adj = adj->succ()) {
                if (!adj->isSource()) continue;
                
                ogdf::edge e = adj->theEdge();
                ogdf::node v2  = adj->twinNode();
                
                if(dfsNum[v]+1 == dfsNum[v2]) {
                    // if(etype[e] == EdgeType::TREE) edgeToAdd=e;
                    if(edgeToAdd!=nullptr) {
                        edgeToAdd = nullptr;
                        break;
                    } 
                    edgeToAdd = e;
                    cntVnext++;
                }
            }
            if(cntVnext == 1 && edgeToAdd != nullptr) {
                result.push_back(edgeToAdd);
            }
        }
        if(dfsNum[v]<dfsNum[z]-1) {
            v = dfsNumInverse[dfsNum[v] + 1];
        } else {
            break;
        }
    }

    return;
}


void FeedbackArcSet::find_feedback_arcs(
    // const ogdf::Graph &Gorig,
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
        ogdf::node u = e->source(), w = e->target();
        ogdf::node uC = orig2copy[u], wC = orig2copy[w];
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
    PROFILE_FUNCTION();

    ogdf::NodeArray<int> comp(this->G);
    int sccs = strongComponents(this->G, comp);

    std::vector<int> size(sccs, 0);
    for (ogdf::node v : this->G.nodes) ++size[comp[v]];

    int trivial = 0, nonTrivial = 0, ntIdx = -1;

    for (int i = 0; i < sccs; ++i) {
        if (size[i] > 1) { ++nonTrivial; ntIdx = i; }
        else ++trivial;
    }

    if (nonTrivial >= 2){
        return {};
    } else if (nonTrivial == 1) {
        // std::vector<ogdf::node> toRemove;
        ogdf::NodeArray<bool> toRemove(this->G, false);
        for (ogdf::node v : this->G.nodes) {
            if (comp[v] != ntIdx) toRemove[v] = true;
        }

        std::vector<ogdf::edge> res;
        this->find_feedback_arcs(res, toRemove);
        return res;

        
    } else {
        std::vector<ogdf::edge> res;
        for(ogdf::edge &e:this->G.edges) res.push_back(e);
        return res;
    }

}