#include <ogdf/basic/graph_generators.h>
#include <ogdf/layered/DfsAcyclicSubgraph.h>
#include <ogdf/fileformats/GraphIO.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/planarity/PlanarizationLayout.h>
#include <ogdf/decomposition/BCTree.h>  
#include <ogdf/decomposition/DynamicSPQRForest.h>
#include <ogdf/decomposition/DynamicSPQRTree.h>
#include <ogdf/augmentation/DfsMakeBiconnected.h>
#include <ogdf/decomposition/StaticSPQRTree.h>
#include <ogdf/decomposition/SPQRTree.h>
#include <ogdf/energybased/FMMMLayout.h>
#include <ogdf/tree/TreeLayout.h>
#include <ogdf/basic/List.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <stack>
#include <cassert>
#include <chrono>
#include <regex>
#include <cassert>


#include "io/graph_io.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"


using namespace ogdf;


static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " -g <graphFile> -o <outputFile> [--gfa]\n";
    std::exit(EXIT_FAILURE);
}


static std::string nextArgOrDie(const std::vector<std::string>& a, std::size_t& i, const char* flag) {
    if (++i >= a.size() || (a[i][0] == '-' && a[i] != "-")) {
        std::cerr << "Error: missing path after " << flag << "\n";
        usage(a[0].c_str());
    }
    return a[i];
}


void readArgs(int argc, char** argv) {
    auto& C = ctx();

    std::vector<std::string> args(argv, argv+argc);

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& s = args[i];

        if (s == "-g") {
            C.graphPath = nextArgOrDie(args, i, "-g");

        } else if (s == "-o") {
            C.outputPath = nextArgOrDie(args, i, "-o");

        } else if (s == "--gfa") {
            C.gfaInput = true;

        } else if (s == "-h" || s == "--help") {
            usage(argv[0]);            // exits

        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(argv[0]);
        }
    }
}




namespace solver {
    struct BlockData {
        std::unique_ptr<ogdf::Graph> Gblk;
        ogdf::NodeArray<ogdf::node> toCc;
        ogdf::NodeArray<ogdf::node> toBlk;
        ogdf::NodeArray<ogdf::node> toOrig;

        std::unique_ptr<ogdf::StaticSPQRTree> spqr;
        std::unordered_map<ogdf::edge, ogdf::edge> skel2tree; // mapping from skeleton virtual edge to tree edge
        ogdf::NodeArray<ogdf::node> parent; // mapping from node to parent in SPQR tree, it is possible since it is rooted, 
                                            // parent of root is nullptr


        ogdf::node bNode {nullptr};


        ogdf::NodeArray<int> inDeg;
        ogdf::NodeArray<int> outDeg;
    };

    struct CcData {
        std::unique_ptr<ogdf::Graph> Gcc;
        ogdf::NodeArray<ogdf::node> toOrig;
        ogdf::NodeArray<ogdf::node> toCopy;
        std::unique_ptr<ogdf::BCTree> bc;
        std::vector<BlockData> blocks;
    };



    void printBlockEdges(std::vector<CcData> &comps) {
        auto& C = ctx();

        std::cout << "Printing block edges:\n";
        std::cout << "----------------------------------------\n";
        for (size_t cid = 0; cid < comps.size(); ++cid) {
            const CcData &cc = comps[cid];

            std::cout << "Component #" << cid << " (" << cc.blocks.size() << "blocks)"  <<  '\n';


            for (size_t bid = 0; bid < cc.blocks.size(); ++bid) {
                const BlockData &blk = cc.blocks[bid];

                std::cout << "  Block #" << bid
                        << "  edges (block-indices): ";

                const Graph &Gb = *blk.Gblk;
                for (edge eB : Gb.edges) {
                    node uB = eB->source();
                    node vB = eB->target();
                    std::cout << uB->index() << "-" << vB->index() << ' ';

                    
                    node uG = cc.toOrig[ blk.toCc[uB] ];
                    node vG = cc.toOrig[ blk.toCc[vB] ];
                    std::cout << '(' << C.node2name[uG] << "-" << C.node2name[vG] << ") ";

                }
                std::cout << '\n';
            }
        }
        std::cout << "----------------------------------------\n";
    }





    enum EdgeType { TREE, BACK, FORWARD, CROSS };



    void run_fas(Graph G, 
        std::vector<edge> &result
        // std::vector<node> &toRemove
    ) {
        // remove nodes that are in trivial SSCs to run only on non-trivial(trivial cannot be FAS)
        // for (node v : toRemove)
        //     G.delNode(v);                                    



        NodeArray<int>  disc(G, 0), finish(G, 0), dfsNum(G, 0);
        NodeArray<node> parent(G, nullptr);
        NodeArray<char> colour(G, 0);
        EdgeArray<EdgeType> etype(G);   
        int time = 0, dfsTime=0;
        std::vector<node> dfsNumInverse(G.maxNodeIndex()+1, nullptr);
        edge singleBackEdge = nullptr;

        
        // Building dfs tree and classifying edges
        std::function<void(node)> dfs1 = [&](node u)
        {
            colour[u] = 1;
            disc[u]  = ++time;
            if(dfsNum[u] == 0) {
                dfsNum[u] = dfsTime;
                dfsNumInverse[dfsTime] = u;
                dfsTime++;
            }
            
            for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
            {
                if (!adj->isSource()) continue;                   
                edge e = adj->theEdge();
                node v = adj->twinNode();
                
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
            finish[u] = ++time;
        };
        
        node root = G.firstNode();
        dfs1(root);
        
        
        
        Graph T;
        NodeArray<node> map(G, nullptr);
        NodeArray<node> mapInverse(T, nullptr);
        for (node v : G.nodes) {
            auto newNode = T.newNode();
            map[v] = newNode;
            mapInverse[newNode] = v;
        }

        
        // for(auto e:G.edges) {
        //     node u = e->source();
        //     node v = e->target();
            
        //     std::cout << dfsNum[u] << " -> " << dfsNum[v] << ": ";
        //     if(etype[e] == TREE) {
        //         std::cout << "Tree edge: " << u->index() << " -> " << v->index() << std::endl;
        //     } else if(etype[e] == BACK) {
        //         std::cout << "Back edge: " << u->index() << " -> " << v->index() << std::endl;
        //     } else if(etype[e] == FORWARD) {
        //         std::cout << "Forward edge: " << u->index() << " -> " << v->index() << std::endl;
        //     } else if(etype[e] == CROSS) {
        //         std::cout << "Cross edge: " << u->index() << " -> " << v->index() << std::endl;
        //     }
        // }
        
        NodeArray<int> backedgeTailSubtreeCount(G, 0);
        int backEdgeCount = 0;



        node y = nullptr;

        for(auto &e:G.edges) {
            node u = e->source();
            node v = e->target();
            
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
        node z = G.firstNode(); 
        {

            std::function<void(node, node)> dfsSubtree = [&](node u, node prev=nullptr) {
                for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;
                    node v = adj->twinNode();
                    if (v == prev) continue;
                    dfsSubtree(v, u);
                    backedgeTailSubtreeCount[u] += backedgeTailSubtreeCount[v];  // accumulate back edges from child
                }
            };

            dfsSubtree(map[root], nullptr);

            // for(auto &u:G.nodes) {
            //     std::cout << "Node " << u->index() << ": back edges in subtree = " << backedgeTailSubtreeCount[u] << std::endl;
            // }   

            std::function<void(node, node)> dfs2 = [&](node u, node prev=nullptr) -> void {
                int c = backedgeTailSubtreeCount[u];
                if (c == backEdgeCount) {
                    z = mapInverse[u];
                    // std::cout << mapInverse[u]->index() << " is a candidate for deepest node in back edges." << std::endl;
                }
                for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
                    if (adj->isSource()) {
                        node v = adj->twinNode();
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

        if(dfsNum[y]>dfsNum[z]) {
            // std::cout << "y > z, no feedback vertices" << std::endl;
            return;
        }

        // removing [y,z] and checking acyclicity
        {
            std::function<bool(node)> inInterval = [&](node u) -> bool {
                return dfsNum[y] <= dfsNum[u] && dfsNum[u] <= dfsNum[z];
            };
            Graph Gp;
            NodeArray<node> map(G, nullptr);
            for (node v : G.nodes)
                map[v] = Gp.newNode();


            for (edge e : G.edges) {
                node u = e->source(), v = e->target();
                // std::cout << u << " -> " << v << std::endl;
                // std::cout << (inInterval(u)) << " " << (inInterval(v)) << std::endl; 
                if(inInterval(u) && inInterval(v) && etype[e]==EdgeType::TREE) continue;
                // std::cout << "added" << std::endl;
                // if(!inInterval(u) && !inInterval(v)) 
                Gp.newEdge(map[u], map[v]);
            }

            GraphIO::drawGraph(Gp, "removed_input"+to_string(G.nodes.size()));

            if(!isAcyclic(Gp)) {
                std::cout << "Graph without interval is not acylic, so there is no feedback set" << std::endl;
                return;
            }
        }


        NodeArray<bool> loop(G, false);
        NodeArray<int> maxi(G, 0);
        {
            NodeArray<int> disc(G), fin(G);
            int time=0;
            std::function<bool(node, node)> isDescendant = [&](node ancestor, node v) -> bool {
                return ancestor != v && disc[ancestor] < disc[v] && fin[v] < fin[ancestor];
            };


            NodeArray<bool> vis(G, false);
            std::function<void(node)> dfsTime = [&](node u) {
                vis[u] = true;
                disc[u] = ++time;
                // std::cout << "at " << u << std::endl;
                for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ())
                    if (adj->isSource()) {            // follow children only
                        node v = adj->twinNode();
                        if (vis[v]) continue;
                        dfsTime(v);
                    }
                fin[u] = ++time;
                
            };
            
            dfsTime(root);
            // return ;

            vis = NodeArray<bool>(G, false);

            std::vector<node> stk;
            stk.reserve(G.nodes.size());
            std::function<void(node)> dfsOrder = [&](node u) {
                vis[u] = true;
                for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (adj->isSource()) {
                        edge e = adj->theEdge();
                        node v = adj->twinNode();
                        if (!vis[v]) dfsOrder(v);
                    }
                }
                stk.push_back(u);
            };

            dfsOrder(root);
            // std::cout << stk.size() << " size "<< std::endl;

            std::reverse(stk.begin(), stk.end());
            while(!stk.empty()) {
                node u = stk.back();
                stk.pop_back();
                // std::cout << "at " << u << std::endl;

                
                // maxi computation
                for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;

                    edge e = adj->theEdge();
                    node v  = adj->twinNode();

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
                        std::cout << dfsNum[v] << std::endl;
                    } else if(dfsNum[v]>dfsNum[z]) {
                        maxi[u] = std::max(maxi[u], maxi[v]);
                        std::cout << maxi[v] << std::endl;
                    }

                }


                // loop computation
                if(isDescendant(z, u)) {
                    loop[u] = true;
                    // std::cout << " it is proper desc of z" << std::endl; 
                } else {
                    for (adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                        if (!adj->isSource()) continue;

                        edge e = adj->theEdge();
                        node v  = adj->twinNode();

                        if(etype[e]!=EdgeType::BACK) {
                            loop[u] |= (dfsNum[v]>dfsNum[z] && loop[v]);
                        }
                    }
                }
            }
        }

        
        
        // init
        node v=y;
        int maxitest=-1;
        for (int i = 0; i < dfsNum[y]; i++) {
            std::cout << dfsNumInverse[i] << std::endl;
            maxitest=std::max(maxitest, maxi[dfsNumInverse[i]]);
        }

        // std::vector<edge> result;
        // step 4
        while(true) {
            
            std::cout << dfsNum[v] << " " << maxitest  << std::endl;

            if(loop[v] == true) {
                break;
            }

            maxitest = std::max(maxitest, maxi[v]);


            if(maxitest<=dfsNum[v]) {
                int cntVnext=0;
                edge edgeToAdd=nullptr;
                for (adjEntry adj = v->firstAdj(); adj; adj = adj->succ()) {
                    if (!adj->isSource()) continue;
                    
                    edge e = adj->theEdge();
                    node v2  = adj->twinNode();
                    
                    // if(v2 == dfsNumInverse[dfsNum[v] + 1]) {
                    
                    if(dfsNum[v]+1 == dfsNum[v2]) {
                        if(etype[e] == EdgeType::TREE) edgeToAdd=e;
                        cntVnext++;
                    }
                    // if(dfsNum[y]<=dfsNum[v2]&& dfsNum[v2]<=dfsNum[z] && etype[e] == EdgeType::TREE) {
                    //     result.push_back(e);
                    // }
                }
                if(cntVnext == 1 && edgeToAdd != nullptr) {
                    result.push_back(edgeToAdd);
                }
                
                // result.push_back({v, dfsNumInverse[dfsNum[v] + 1]});
            }
            if(dfsNum[v]<dfsNum[z]-1) {
                v = dfsNumInverse[dfsNum[v] + 1];
            } else {
                break;
            }
        }


        std::cout << "Feedback edges: ";
        for(auto &w:result) {
            std::cout << w->source() << " -> " << w->target() << ", ";
        }
        std::cout << std::endl;

        return;
    }


    // Given a graph G and possible to remove nodes(to make it trivial SSCs),
    // find all edges that if removed, would make the graph acyclic.
    // O(n + m) time complexity.

    void find_feedback_arcs(const Graph &Gorig,           // caller’s graph
                        std::vector<edge> &result,    // ← edges from Gorig!
                        const std::vector<node> &toRemove)
    {
        Graph G;
        NodeArray<node> orig2copy (Gorig, nullptr);
        NodeArray<node> copy2orig (G,     nullptr);
        EdgeArray<edge> copy2origE(G,     nullptr);

        for (node v : Gorig.nodes) {
            node w = G.newNode();
            orig2copy[v] = w;
            copy2orig[w] = v;
        }
        for (edge e : Gorig.edges) {
            edge f = G.newEdge(orig2copy[e->source()], orig2copy[e->target()]);
            copy2origE[f] = e;
        }

        for (node vOrig : toRemove)
            if (orig2copy[vOrig]) G.delNode(orig2copy[vOrig]);

        std::vector<edge> copyResult;
        run_fas(G, copyResult);

        result.clear();
        result.reserve(copyResult.size());
        for (edge f : copyResult)
            result.push_back(copy2origE[f]);
    }


    



    // add a new superbubble (if possible)
    void addSuperbubble(ogdf::node source, ogdf::node sink) {
        auto& C = ctx();

        if(C.isEntry[source] || C.isExit[sink]) {
            std::cerr << ("Superbubble already exists for source %s and sink %s", C.node2name[source].c_str(), C.node2name[sink].c_str());
            return;
        }
        C.isEntry[source] = true;
        C.isExit[sink] = true;
        C.superbubbles.emplace_back(source, sink);
        std::cout << "added " << ctx().node2name[source] << "," << ctx().node2name[sink] << std::endl; 

    }




    // build whole decomposition of graph
    // Graph -> weakly connected components -> biconnected components + spqr tree(if possible)
    static std::vector<CcData> buildDecomposition(Graph &G) {
        logger::info("Building graph decomposition..");
        NodeArray<int> compIdx(G);
        int nCC = connectedComponents(G, compIdx);

        std::vector<std::vector<node>> bucket(nCC);
        for (node v : G.nodes) bucket[compIdx[v]].push_back(v);

        std::vector<CcData> out;
        out.reserve(nCC);

        logger::info(to_string(nCC) + " connected components");
        
        for (int cid = 0; cid < nCC; ++cid) {
            std::cout << cid << std::endl;
            auto Gcc = std::make_unique<Graph>();
            NodeArray<node> toOrig(*Gcc, nullptr);
            NodeArray<node> toCopy(G, nullptr);

            for (node vOrig : bucket[cid]) {
                node vCopy       = Gcc->newNode();
                toCopy[vOrig]    = vCopy;
                toOrig[vCopy]    = vOrig;
            }

            for (edge e : G.edges) {
                node u = e->source(), v = e->target();
                if (compIdx[u] != cid || compIdx[v] != cid) continue;
                Gcc->newEdge(toCopy[u], toCopy[v]);
            }

            std::unique_ptr<BCTree> bc(new BCTree(*Gcc));

            std::vector<BlockData> blkVec;


            std::cout << "aaa" << std::endl;

            for (node b : bc->bcTree().nodes) {
                std::cout << b << "/" << bc->bcTree().nodes.size() << std::endl;
                if (bc->typeOfBNode(b) == BCTree::BNodeType::BComp) {
                    std::unordered_set<ogdf::node> nodeSet;

                    for (ogdf::edge hE : bc->hEdges(b)) {
                        ogdf::edge eCc = bc->original(hE);
                        nodeSet.insert(eCc->source());
                        nodeSet.insert(eCc->target());
                    }

                    std::unique_ptr<ogdf::Graph>  Gblk(new ogdf::Graph);
                    ogdf::NodeArray<ogdf::node>   toCcBlk(*Gblk, nullptr);
                    ogdf::NodeArray<ogdf::node>   toBlk(*Gcc, nullptr);
                    ogdf::NodeArray<ogdf::node>   toOrigBlk(*Gblk, nullptr); 




                    for (ogdf::node vCc : nodeSet) {
                        ogdf::node vB = Gblk->newNode();
                        toBlk[vCc] = vB;
                        toCcBlk[vB] = vCc;
                        toOrigBlk[vB] = toOrig[vCc];
                    }

                    for (ogdf::edge hE : bc->hEdges(b)) {
                        ogdf::edge eCc = bc->original(hE);
                        Gblk->newEdge(toBlk[eCc->source()], toBlk[eCc->target()]);
                    }


                    std::unique_ptr<ogdf::StaticSPQRTree> spqr;
                    if(Gblk->numberOfNodes()>=3) {
                        spqr.reset(new ogdf::StaticSPQRTree(*Gblk));
                    }


                    ogdf::NodeArray<int> inDeg (*Gblk, 0);
                    ogdf::NodeArray<int> outDeg(*Gblk, 0);

                    for (ogdf::edge e : Gblk->edges) {
                        ++outDeg[e->source()];
                        ++inDeg [e->target()];
                    }

                    
                    std::unordered_map<ogdf::edge, ogdf::edge> localMap;
                    ogdf::NodeArray<ogdf::node> parentTmp;

                    if (spqr) {
                        const auto &T = spqr->tree();
                        for (ogdf::edge te : T.edges) {
                            ogdf::edge eSrc = spqr->skeletonEdgeSrc(te);
                            ogdf::edge eTgt = spqr->skeletonEdgeTgt(te);

                            localMap[eSrc] = te;
                            localMap[eTgt] = te;
                        }
                        
                        parentTmp.init(T, nullptr);

                        ogdf::node root = spqr->rootNode();
                        std::stack<ogdf::node> S;
                        parentTmp[root] = root;
                        S.push(root);

                        while (!S.empty()) {
                            ogdf::node u = S.top(); S.pop();
                            for (ogdf::adjEntry adj : u->adjEntries) {
                                ogdf::node v = adj->twinNode();
                                if (parentTmp[v] == nullptr) {
                                    parentTmp[v] = u;
                                    S.push(v);
                                }
                            }
                        }
                    }


                    
                    blkVec.push_back({ std::move(Gblk),
                                    std::move(toCcBlk),
                                    std::move(toBlk),
                                    std::move(toOrigBlk),
                                    std::move(spqr),
                                    std::move(localMap),
                                    std::move(parentTmp),
                                    std::move(b),
                                    std::move(inDeg),
                                    std::move(outDeg) });
                }
            }

            out.push_back({ std::move(Gcc),
                            std::move(toOrig),
                            std::move(toCopy),
                            std::move(bc),
                            std::move(blkVec) });
        }

        logger::info("Graph decomposition finished");


        return out;
    }



    // Check if whole block node is superbubble
    // void checkWholeBlockSuperbubble(const BlockData &blk, const CcData &cc) {
    //     auto& C = ctx();

    //     std::vector<ogdf::node> sources, sinks;

    //     const NodeArray<int> &globIn  = C.inDeg;   // global arrays
    //     const NodeArray<int> &globOut = C.outDeg;


    //     for (node vB : blk.Gblk->nodes) {
    //         node vG  = blk.toOrig[vB]; // global node

    //         int inLocal  = blk.inDeg [vB];
    //         int outLocal = blk.outDeg[vB];

    //         int inGlobal  = globIn [vG];
    //         int outGlobal = globOut[vG];

    //         if (inLocal == 0 && outLocal == outGlobal) sources.push_back(vB);
    //         if (outLocal == 0 && inLocal == inGlobal) sinks.push_back(vB);

    //         if(inLocal != 0 && outLocal != 0 && (outLocal != outGlobal || inLocal != inGlobal)) return;
    //     }

    //     if(sources.size() != 1 || sinks.size() != 1) return;
        

    //     ogdf::node source = sources[0];
    //     ogdf::node sink   = sinks[0];

    //     ogdf::node sourceG = blk.toOrig[source];
    //     ogdf::node sinkG   = blk.toOrig[sink];

    //     if(isAcyclic(*blk.Gblk)) {
    //         addSuperbubble(sourceG, sinkG);
    //     }

    // }

    void checkBlockByCutVertices(const BlockData &blk, const CcData &cc)    
{
    auto &C      = ctx();
    const Graph &G = *blk.Gblk;

    node src=nullptr, snk=nullptr;

    /* ---------- 1. classify every vertex -------------------------------- */
    for (node v : G.nodes) {
        node vG   = blk.toOrig[v];
        int inL   = blk.inDeg [v], outL = blk.outDeg[v];
        int inG   = C.inDeg  [vG], outG = C.outDeg[vG];

        bool isSrc = (inL  == 0 && outL == outG);
        bool isSnk = (outL == 0 && inL == inG);

        if (isSrc ^ isSnk) {               // pure role
            if (isSrc) { if (src){ std::cout<<" second src → reject\n"; return;} src=v; }
            else       { if (snk){ std::cout<<" second snk → reject\n"; return;} snk=v; }
        }
        else if (!(inL == inG && outL == outG)) {
            std::cout << " leak at v="<<v->index()<<" → reject\n";
            return;                         // rule ② failed
        }
    }

    if (!src || !snk) { std::cout<<" need 1 src & 1 snk → reject\n"; return; }

    /* ---------- 2. acyclicity ------------------------------------------- */
    if (!isAcyclic(G)) { std::cout<<" cycle inside block → reject\n"; return; }

    /* ---------- 3. reachability ----------------------------------------- */
    NodeArray<bool> vis(G,false); std::stack<node> S; vis[src]=true; S.push(src);
    bool reach=false;
    while(!S.empty() && !reach){
        node u=S.top();S.pop();
        for(adjEntry a=u->firstAdj();a;a=a->succ())
            if(a->isSource()){
                node v=a->twinNode();
                if(!vis[v]){ if(v==snk){reach=true;break;}
                             vis[v]=true; S.push(v);}
            }
    }
    if(!reach){ std::cout<<" src cannot reach snk → reject\n"; return; }

    /* ---------- 4. success ---------------------------------------------- */
    node srcG = blk.toOrig[src], snkG = blk.toOrig[snk];
    if (C.isEntry[srcG] || C.isExit[snkG]) { std::cout<<" already reported\n"; return; }

    std::cout<<" superbubble "<<C.node2name[srcG]<<" → "<<C.node2name[snkG]<<'\n';
    addSuperbubble(srcG,snkG);                // marks isEntry/isExit
}



    // Part that solves for block node
    namespace SPQRsolve {
        struct EdgeDPState {
            node s{nullptr};      
            node t{nullptr};

            int localOutS{0};
            int localInT{0};
            int localOutT{0};
            int localInS{0};

            bool globalSourceSink{false}; 

            bool directST{false};
            bool directTS{false};

            bool hasLeakage{false};

            bool acyclic{true};

            int getDirection() const {
                if(acyclic && !globalSourceSink && localOutS>0 && localInT>0) return 1; // s -> t
                if(acyclic && !globalSourceSink && localOutT>0 && localInS>0) return -1; // t -> s
                return 0; // no direction ?
            }
        };

        struct NodeDPState {
            int outgoingCyclesCount{0}; 
            node lastCycleNode{nullptr}; 
            int outgoingSourceSinkCount{0};
            node lastSourceSinkNode{nullptr};
            int outgoingLeakageCount{0};
            node lastLeakageNode{nullptr};
        };

        // pair of dp states for each edge for both directions
        struct EdgeDP {
            EdgeDPState down;   // value valid in  parent -> child  direction
            EdgeDPState up;     // value valid in  child -> parent direction
        };


        void printAllStates(ogdf::EdgeArray<EdgeDP> &edge_dp, ogdf::NodeArray<NodeDPState> &node_dp,  const Graph &T) {
            auto& C = ctx();


            std::cout << "Edge dp states:" << std::endl;
            for(auto &e:T.edges) {
                {
                    EdgeDPState state = edge_dp[e].down;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->source() << " -> " << e->target() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        

                        std::cout << std::endl;
                        // std::cout << "hasGlobalSourceSink = " << state.hasGlobalSourceSink << ", ";
                        // std::cout << "localSOutDeg = " << state.localSOutDeg << ", ";
                        // std::cout << "localSInDeg = " << state.localSInDeg << ", ";
                        // std::cout << "localTOutDeg = " << state.localTOutDeg << ", ";
                        // std::cout << "localTInDeg = " << state.localTInDeg << ", ";
                        // std::cout << "directSTCount = " << state.directSTCount << ", ";
                        // std::cout << "directTSCount = " << state.directTSCount << ", ";
                        // std::cout << "hasLeakage = " << (state.hasLeakage) << ", ";
                    }
                }

                {
                    EdgeDPState state = edge_dp[e].up;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->target() << " -> " << e->source() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        

                        std::cout << std::endl;
                        // std::cout << "hasGlobalSourceSink = " << state.hasGlobalSourceSink << ", ";
                        // std::cout << "localSOutDeg = " << state.localSOutDeg << ", ";
                        // std::cout << "localSInDeg = " << state.localSInDeg << ", ";
                        // std::cout << "localTOutDeg = " << state.localTOutDeg << ", ";
                        // std::cout << "localTInDeg = " << state.localTInDeg << ", ";
                        // std::cout << "directSTCount = " << state.directSTCount << ", ";
                        // std::cout << "directTSCount = " << state.directTSCount << ", ";
                        // std::cout << "hasLeakage = " << (state.hasLeakage) << ", ";
                    }
                }
            }

            std::cout << "Node dp states: " << std::endl;
            for(node v : T.nodes) {
                std::cout << "Node " << v->index() << ", ";
                std::cout << "outgoingCyclesCount: " << node_dp[v].outgoingCyclesCount << ", ";
                std::cout << "outgoingLeakageCount: " << node_dp[v].outgoingLeakageCount << ", ";
                std::cout << "outgoingSourceSinkCount: " << node_dp[v].outgoingSourceSinkCount << ", ";

                std::cout << std::endl;
                
            }
        }

        void dfsSPQR_order(
            SPQRTree &spqr,
            std::vector<ogdf::edge> &edge_order, // order of edges to process
            std::vector<ogdf::node> &nodeOrder,
            node curr = nullptr,
            node parent = nullptr,
            edge e = nullptr 
        ) {
            if(curr == nullptr) {
                curr = spqr.rootNode();
                parent = curr;
                dfsSPQR_order(spqr, edge_order, nodeOrder, curr, parent);
                return;
            }
            nodeOrder.push_back(curr);
            for (adjEntry adj : curr->adjEntries) {
                node child = adj->twinNode();
                if (child == parent) continue;
                dfsSPQR_order(spqr, edge_order, nodeOrder, child, curr, adj->theEdge());
            }
            if(curr!=parent) edge_order.push_back(e);
        }



        // process edge in the direction of parent to child
        // Computing A->B (curr_edge)
        void processEdge(ogdf::edge curr_edge, ogdf::EdgeArray<EdgeDP> &dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, const BlockData &blk) {
            auto& C = ctx();

            const ogdf::NodeArray<int> &globIn  = C.inDeg;
            const ogdf::NodeArray<int> &globOut = C.outDeg;
                        
            EdgeDPState &state = dp[curr_edge].down;
            EdgeDPState &back_state = dp[curr_edge].up;
            
            const StaticSPQRTree &spqr = *blk.spqr;
                        
            ogdf::node A = curr_edge->source();
            ogdf::node B = curr_edge->target();
            
            std::cout << curr_edge->source() << " " << curr_edge->target() << std::endl;
            std::cout << "Processing edge: " << A->index() << " -> " << B->index() << '\n';
            
            // std::tie(state.s, state.t) = getPoles(spqr, cc, A, B);
            // back_state.s = state.s;
            // back_state.t = state.t;
            
            // std::cout << "Set pole of " << A << ">" << B << " as (" << C.node2name[state.s] << "," << C.node2name[state.t] << ")" << std::endl;

            
            state.localOutS = 0;
            state.localInT  = 0;
            state.localOutT = 0;
            state.localInS  = 0;
            
            const Skeleton &skel = spqr.skeleton(B);
            const Graph &skelGraph = skel.getGraph();

            // EdgeArray<edge> skel2tree(skelGraph, nullptr);
            // for (edge te : spqr.tree().edges) {
            //     if (te->source() == B) {
            //         edge ve = spqr.skeletonEdgeSrc(te);
            //         skel2tree[ve] = te;
            //     }
            //     else if (te->target() == B) {
            //         edge ve = spqr.skeletonEdgeTgt(te);
            //         skel2tree[ve] = te;
            //     }
            // }


            
            // Building new graph with correct orientation of virtual edges
            Graph newGraph;

            NodeArray<node> skelToNew(skelGraph, nullptr);
            for (node v : skelGraph.nodes) skelToNew[v] = newGraph.newNode();
            NodeArray<node> newToSkel(newGraph, nullptr);
            for (node v : skelGraph.nodes) newToSkel[skelToNew[v]] = v;


            ogdf::NodeArray<ogdf::node> compToSkel(*blk.Gblk, nullptr);
            for (ogdf::node h : skelGraph.nodes) {
                ogdf::node vCc = skel.original(h);   // component node
                compToSkel[vCc] = h;                 // save reverse link
            }

            
            NodeArray<int> localInDeg(newGraph, 0), localOutDeg(newGraph, 0);




            auto mapGlobalToNew = [&](ogdf::node vG) -> ogdf::node {
                // global -> component
                ogdf::node vComp = cc.toCopy[vG];
                if (!vComp) return nullptr;

                // component -> block
                ogdf::node vBlk  = blk.toBlk[vComp];
                if (!vBlk)  return nullptr;

                // block -> skeleton
                ogdf::node vSkel = compToSkel[vBlk];
                if (!vSkel) return nullptr;

                return skelToNew[vSkel];
            };

            auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node {
                if (!vN) return nullptr;

                ogdf::node vSkel = newToSkel[vN];
                if (!vSkel) return nullptr;

                ogdf::node vBlk  = skel.original(vSkel);
                if (!vBlk) return nullptr;

                ogdf::node vCc   = blk.toCc[vBlk];
                if (!vCc) return nullptr;

                return cc.toOrig[vCc];
            };




            auto printDegrees = [&]() {
                for(node vN:newGraph.nodes) {
                    node vG = mapNewToGlobal(vN);

                    std::cout << C.node2name[vG] << ":    out: " << localOutDeg[vN] << ", in: " << localInDeg[vN] << std::endl;  
                }
            };





            for(edge e : skelGraph.edges) {
                printDegrees();
                node u = e->source();
                node v = e->target();

                node nU = skelToNew[u];
                node nV = skelToNew[v];
                

                if(!skel.isVirtual(e)) {
                    std::cout << "real " << C.node2name[mapNewToGlobal(nU)] << " -> " << C.node2name[mapNewToGlobal(nV)] << std::endl;

                    newGraph.newEdge(nU, nV);
                    localOutDeg[nU]++;
                    localInDeg[nV]++;

                    std::cout << "added" << std::endl;
                    continue;
                }
                
                auto D = skel.twinTreeNode(e);
                

                if(D == A) {
                    std::cout << "virtual edge to parent" << std::endl;
                    ogdf::node vCc = skel.original(u);
                    ogdf::node uCc = skel.original(v);
                    
                    ogdf::node vG  = blk.toOrig[vCc];
                    ogdf::node uG  = blk.toOrig[uCc];

                    state.s = back_state.s = vG;
                    state.t = back_state.t = uG;


                    std::cout << "found" << std::endl;

                    continue;
                }

                // ogdf::edge treeE = skel2tree[e];
                // assert(treeE);

                edge treeE = blk.skel2tree.at(e);
                OGDF_ASSERT(treeE != nullptr);



                const EdgeDPState child = dp[treeE].down;
                int dir = child.getDirection();

                ogdf::node nS = mapGlobalToNew(child.s);
                ogdf::node nT = mapGlobalToNew(child.t);

                std::cout << C.node2name[child.s] << " " << C.node2name[child.t] << ": " << dir << std::endl;

                

                if(dir==1) {
                    // std::cout << "added " << C.node2name[mapNewToGlobal(nS)] << " -> " << C.node2name[mapNewToGlobal(nT)] << std::endl;

                    newGraph.newEdge(nS, nT);

                    // localOutDeg[nS]+=child.localOutS; 
                    // localInDeg[nS]+=child.localInS;

                    // localOutDeg[nT]+=child.localOutT;
                    // localInDeg[nT]+=child.localInT;
                } else if(dir==-1) {
                    // std::cout << "added " << C.node2name[mapNewToGlobal(nT)] << " -> " << C.node2name[mapNewToGlobal(nS)] << std::endl;
                    newGraph.newEdge(nT, nS);
                } 
                // else {
                //     newGraph.newEdge(nS, nT);
                //     newGraph.newEdge(nT, nS);
                // }

                if(nS == nU && nT == nV) {
                    localOutDeg[nS]+=child.localOutS; 
                    localInDeg[nS]+=child.localInS;

                    localOutDeg[nT]+=child.localOutT;
                    localInDeg[nT]+=child.localInT;
                } else {
                    localOutDeg[nT]+=child.localOutT; 
                    localInDeg[nT]+=child.localInT;

                    localOutDeg[nS]+=child.localOutS;
                    localInDeg[nS]+=child.localInS;
                }



                state.acyclic &= child.acyclic;
                state.globalSourceSink |= child.globalSourceSink;
                state.hasLeakage |= child.hasLeakage;
            }

            printDegrees();

            GraphIO::drawGraph(newGraph, "newGraph"+to_string(A->index())+">"+to_string(B->index()));

            // for(node &v:newGraph.nodes) {
            //     std::cout << v << ":   in: " << localInDeg[v] << ", out: " << localOutDeg[v] << std::endl;
            // }


            // Direct ST/TS computation(only happens in P nodes)
            if(spqr.typeOf(B) == SPQRTree::NodeType::PNode) {
                for(edge e : skelGraph.edges) {
                    if(skel.isVirtual(e)) continue;
                    node u = e->source();
                    node v = e->target();

                    node nU = skelToNew[u];
                    node nV = skelToNew[v];

                    if(mapGlobalToNew(state.s) == nU && mapGlobalToNew(state.t) == nV) {
                        state.directST = true;
                    } else if(mapGlobalToNew(state.s) == nV && mapGlobalToNew(state.t) == nU) {
                        state.directTS = true;
                    } else {
                        assert(false);
                    }
                }
            }


            for (ogdf::node vN : newGraph.nodes) {
                ogdf::node vG  = mapNewToGlobal(vN);
                assert(vN == mapGlobalToNew(vG));

                if (vG == state.s || vG == state.t)
                    continue;

                
                if(globIn[vG] != localInDeg[vN] || globOut[vG] != localOutDeg[vN]) {
                    state.hasLeakage = true;
                }

                if (globIn[vG] == 0 || globOut[vG] == 0) {
                    state.globalSourceSink = true;
                }
            }





            state.localInS = localInDeg[mapGlobalToNew(state.s)];
            state.localOutS = localOutDeg[mapGlobalToNew(state.s)];

            state.localInT = localInDeg[mapGlobalToNew(state.t)];
            state.localOutT = localOutDeg[mapGlobalToNew(state.t)];

            
            if(state.acyclic) state.acyclic &= isAcyclic(newGraph);


            if(!state.acyclic) {
                node_dp[A].outgoingCyclesCount++;
                node_dp[A].lastCycleNode = B;
            }

            if(state.globalSourceSink) {
                node_dp[A].outgoingSourceSinkCount++;
                node_dp[A].lastSourceSinkNode = B;
            }

            if(state.hasLeakage) {
                node_dp[A].outgoingLeakageCount++;
                node_dp[A].lastLeakageNode = B;
            }
        }


        void processNode(node curr_node, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, const BlockData &blk) {
            auto& C = ctx();

            // std::cout << "Processing node " << curr_node << std::endl;
            // logger::info("Processing node {}", curr_node->index());
            // logger::flush();

            const ogdf::NodeArray<int> &globIn  = C.inDeg;
            const ogdf::NodeArray<int> &globOut = C.outDeg;

            ogdf::node A = curr_node;
            
            const Graph &T = blk.spqr->tree();
            
            NodeDPState curr_state = node_dp[A]; 

            const StaticSPQRTree &spqr = *blk.spqr;
                        

            const Skeleton &skel = spqr.skeleton(A);
            const Graph &skelGraph = skel.getGraph();

            
            // Building new graph with correct orientation of virtual edges
            Graph newGraph;

            NodeArray<node> skelToNew(skelGraph, nullptr);
            for (node v : skelGraph.nodes) skelToNew[v] = newGraph.newNode();
            NodeArray<node> newToSkel(newGraph, nullptr);
            for (node v : skelGraph.nodes) newToSkel[skelToNew[v]] = v;


            ogdf::NodeArray<ogdf::node> compToSkel(*blk.Gblk, nullptr);
            for (ogdf::node h : skelGraph.nodes) {
                ogdf::node vCc = skel.original(h);   // component node
                compToSkel[vCc] = h;                 // save reverse link
            }

            
            NodeArray<int> localInDeg(newGraph, 0), localOutDeg(newGraph, 0);
            
            NodeArray<bool> isSourceSink(newGraph, false);
            int localSourceSinkCount = 0;

            NodeArray<bool> isLeaking(newGraph, false);
            int localLeakageCount = 0;

            EdgeArray<bool> isVirtual(newGraph, false);
            EdgeArray<EdgeDPState*> edgeToDp(newGraph, nullptr);
            EdgeArray<EdgeDPState*> edgeToDpR(newGraph, nullptr);
            EdgeArray<node> edgeChild(newGraph, nullptr);




            auto mapGlobalToNew = [&](ogdf::node vG) -> ogdf::node {
                // global -> component
                ogdf::node vComp = cc.toCopy[vG];
                if (!vComp) return nullptr;
                // component -> block
                ogdf::node vBlk  = blk.toBlk[vComp];
                if (!vBlk)  return nullptr;
                // block -> skeleton
                ogdf::node vSkel = compToSkel[vBlk];
                if (!vSkel) return nullptr;

                return skelToNew[vSkel];
            };

            auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node {
                if (!vN) return nullptr;
                ogdf::node vSkel = newToSkel[vN];
                if (!vSkel) return nullptr;
                ogdf::node vBlk  = skel.original(vSkel);
                if (!vBlk) return nullptr;
                ogdf::node vCc   = blk.toCc[vBlk];
                if (!vCc) return nullptr;
                return cc.toOrig[vCc];
            };




            auto printDegrees = [&]() {
                for(node vN:newGraph.nodes) {
                    node vG = mapNewToGlobal(vN);

                    // std::cout << C.node2name[vG] << ":    out: " << localOutDeg[vN] << ", in: " << localInDeg[vN] << std::endl;  
                }
            };

            // std::vector<edge> virtualEdges;


            // Building new graph
            for(edge e : skelGraph.edges) {
                node u = e->source();
                node v = e->target();

                node nU = skelToNew[u];
                node nV = skelToNew[v];
                

                if(!skel.isVirtual(e)) {
                    // std::cout << "real " << C.node2name[mapNewToGlobal(nU)] << " -> " << C.node2name[mapNewToGlobal(nV)] << std::endl;

                    auto newEdge = newGraph.newEdge(nU, nV);

                    isVirtual[newEdge] = false;

                    localOutDeg[nU]++;
                    localInDeg[nV]++;

                    // std::cout << "added" << std::endl;
                    continue;
                }
                
                auto B = skel.twinTreeNode(e);
                
                edge treeE = blk.skel2tree.at(e);
                OGDF_ASSERT(treeE != nullptr);



                EdgeDPState *child = (B == blk.parent(A) ? &edge_dp[treeE].up : &edge_dp[treeE].down);
                EdgeDPState *edgeToUpdate = (B == blk.parent(A) ? &edge_dp[treeE].down : &edge_dp[treeE].up);
                int dir = child->getDirection();

                ogdf::node nS = mapGlobalToNew(child->s);
                ogdf::node nT = mapGlobalToNew(child->t);

                // std::cout << C.node2name[child.s] << " " << C.node2name[child.t] << ": " << dir << std::endl;

                edge newEdge = nullptr;

                if(dir==1 || dir == 0) {
                    // std::cout << "added " << C.node2name[mapNewToGlobal(nS)] << " -> " << C.node2name[mapNewToGlobal(nT)] << std::endl;

                    newEdge = newGraph.newEdge(nS, nT);
                    
                    isVirtual[newEdge] = true;
                    // edgeToDp[newEdge] = &child;

                    edgeToDp[newEdge] = edgeToUpdate;
                    edgeToDpR[newEdge] = child;
                    edgeChild[newEdge] = B;
                    // localOutDeg[nS]+=child.localOutS; 
                    // localInDeg[nS]+=child.localInS;
                    
                    // localOutDeg[nT]+=child.localOutT;
                    // localInDeg[nT]+=child.localInT;
                } else if(dir==-1) {
                    // std::cout << "added " << C.node2name[mapNewToGlobal(nT)] << " -> " << C.node2name[mapNewToGlobal(nS)] << std::endl;
                    newEdge = newGraph.newEdge(nT, nS);
                    
                    isVirtual[newEdge] = true;
                    // edgeToDp[newEdge] = &child;
                    edgeToDpR[newEdge] = child;
                    edgeToDp[newEdge] = edgeToUpdate;
                    edgeChild[newEdge] = B;

                    
                } else {
                    // std::cout << "edge does not have direction, only adding s->t" << std::endl;

                    newEdge = newGraph.newEdge(nS, nT);
                    isVirtual[newEdge] = true;
                    edgeChild[newEdge] = B;
                    edgeToDpR[newEdge] = child;

                    
                    // edgeToDp[newEdge] = &child;
                    edgeToDp[newEdge] = edgeToUpdate;

                }

                if(nS == nU && nT == nV) {
                    localOutDeg[nS]+=child->localOutS; 
                    localInDeg[nS]+=child->localInS;

                    localOutDeg[nT]+=child->localOutT;
                    localInDeg[nT]+=child->localInT;
                } else {
                    localOutDeg[nT]+=child->localOutT; 
                    localInDeg[nT]+=child->localInT;

                    localOutDeg[nS]+=child->localOutS;
                    localInDeg[nS]+=child->localInS;
                }


            }

        


            // printDegrees();

            // std::cout << "curr_node: " << curr_node << std::endl;
            // GraphIO::drawGraph(newGraph, "newGraph"+to_string(A->index()));


            for(node v : newGraph.nodes) {
                node vG = mapNewToGlobal(v);
                if(C.inDeg(vG) == 0 || C.outDeg(vG) == 0) {
                    localSourceSinkCount++;
                    isSourceSink[v] = true;
                }

                if(C.inDeg(vG) != localInDeg[v] || C.outDeg(vG) != localOutDeg[v]) {
                    localLeakageCount++;
                    isLeaking[v] = true;
                }
            }


            // calculating ingoing dp states of direct st and ts edges in P node
            if (spqr.typeOf(A) == StaticSPQRTree::NodeType::PNode) {
                node pole0Blk = nullptr, pole1Blk = nullptr;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end()) pole0Blk = skel.original(*it++);
                    if (it != skelGraph.nodes.end()) pole1Blk = skel.original(*it);
                }

                if (!pole0Blk || !pole1Blk)
                    return;

                node gPole0 = cc.toOrig[blk.toCc[pole0Blk]];
                node gPole1 = cc.toOrig[blk.toCc[pole1Blk]];


                int cnt01 = 0, cnt10 = 0;
                for (edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e))
                    {
                        node uG = cc.toOrig[blk.toCc[skel.original(e->source())]];
                        node vG = cc.toOrig[blk.toCc[skel.original(e->target())]];
                        if (uG == gPole0 && vG == gPole1) ++cnt01;
                        else if (uG == gPole1 && vG == gPole0) ++cnt10;
                    }
                }


                for (edge e : skelGraph.edges) {
                    if (skel.isVirtual(e))
                    {
                        node  B = skel.twinTreeNode(e);
                        edge  treeE = blk.skel2tree.at(e);

                        SPQRsolve::EdgeDPState &st = 
                            (B == blk.parent(A) ? edge_dp[treeE].down
                            : edge_dp[treeE].up);

                        if (st.s == gPole0 && st.t == gPole1) {
                            st.directST |= (cnt01 > 0);
                            st.directTS |= (cnt10 > 0);
                        }
                        else if (st.s == gPole1 && st.t == gPole0) {
                            st.directST |= (cnt10 > 0);
                            st.directTS |= (cnt01 > 0);
                        }
                    }
                }
            } 
    
            // std::cout << "Outgoing cycles: " << node_dp[curr_node].outgoingCyclesCount << std::endl;




            // std::cout << "EDGES COUNT BEFORE: " << newGraph.edges.size() << std::endl;

            // std::cout << "Computing ingoing acyclcity.. " << std::endl;
            if(curr_state.outgoingCyclesCount>=2) {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;
                    if(edgeToDp[e]->acyclic) {
                        node_dp[edgeChild[e]].outgoingCyclesCount++;
                        node_dp[edgeChild[e]].lastCycleNode = curr_node;
                    }
                    edgeToDp[e]->acyclic &= false;
                }
            } else if(node_dp[curr_node].outgoingCyclesCount == 1) {

                std::vector<edge> virt;
                for (edge e : newGraph.edges)
                    if (isVirtual[e])
                        virt.push_back(e);

                for (edge e : virt) {
                    if(edgeChild[e] != curr_state.lastCycleNode) {
                        if(edgeToDp[e]->acyclic) {
                            node_dp[edgeChild[e]].outgoingCyclesCount++;
                            node_dp[edgeChild[e]].lastCycleNode = curr_node;
                        }
                        edgeToDp[e]->acyclic &= false;
                    } else {                        
                        node  u   = e->source();
                        node  v   = e->target();
                        auto *st  = edgeToDp[e];
                        auto *ts  = edgeToDpR[e];
                        auto *child = edgeChild[e];
                        bool  acyclic = false;

                        // std::cout << "removing " << u << " -> " << v << std::endl;
                        newGraph.delEdge(e);
                        acyclic = isAcyclic(newGraph);

                        edge eRest = newGraph.newEdge(u, v);
                        isVirtual[eRest] = true;
                        edgeToDp [eRest] = st;
                        edgeToDpR[eRest] = ts;
                        edgeChild[eRest] = child;
                        // std::cout << 123321 << std::endl;
            
                        if(edgeToDp[eRest]->acyclic && !acyclic) {
                            node_dp[edgeChild[eRest]].outgoingCyclesCount++;
                            node_dp[edgeChild[eRest]].lastCycleNode = curr_node;
                        }

                        edgeToDp[eRest]->acyclic &= acyclic;
                    }
                }

            } else {
                NodeArray<int> comp(newGraph);
                int sccs = strongComponents(newGraph, comp);

                std::vector<int> size(sccs, 0);
                for (node v : newGraph.nodes) ++size[comp[v]];

                int trivial = 0, nonTrivial = 0, ntIdx = -1;

                for (int i = 0; i < sccs; ++i) {
                    if (size[i] > 1) { ++nonTrivial; ntIdx = i; }
                    else ++trivial;
                }

                // std::cout << trivial << ", " << nonTrivial << std::endl;

                if (nonTrivial >= 2){
                    for (edge e : newGraph.edges) {
                        if(!isVirtual[e]) continue;
                        if(edgeToDp[e]->acyclic) {
                            node_dp[edgeChild[e]].outgoingCyclesCount++;
                            node_dp[edgeChild[e]].lastCycleNode = curr_node;
                        }

                        edgeToDp[e]->acyclic &= false;
                    }
                } else if (nonTrivial == 1) {
                    std::vector<node> toRemove;
                    for (node v : newGraph.nodes)
                        if (comp[v] != ntIdx) toRemove.push_back(v);

                    std::vector<edge> fas;
                    find_feedback_arcs(newGraph, fas, toRemove);

                    // std::cout << "Fas size: " << fas.size() << std::endl;
                    // std::cout << "fas: " << std::endl;
                    for(edge e:fas) {
                        std::cout << e << std::endl;
                    }

                    EdgeArray<bool> isFas(newGraph, 0);
                    for (edge e : fas) isFas[e] = true;

                    for (edge e : newGraph.edges) {
                        // std::cout << e->source() << " -> " << e->target() << std::endl;
                        if (!isVirtual[e]) continue;


                        if(edgeToDp[e]->acyclic && !isFas[e]) {
                            node_dp[edgeChild[e]].outgoingCyclesCount++;
                            node_dp[edgeChild[e]].lastCycleNode = curr_node;
                        }

                        edgeToDp[e]->acyclic &= isFas[e];
                    }
                }
            }



            std::cout << "Computing ingoing sources/sinks.. " << std::endl;
            // computing global sources/sinks
            if(curr_state.outgoingSourceSinkCount >= 2) {
                // all ingoing have source
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;

                    if(!edgeToDp[e]->globalSourceSink) {
                        node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                        node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                    }


                    edgeToDp[e]->globalSourceSink |= true;
                }
            } else if(curr_state.outgoingSourceSinkCount == 1) {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;
                    if(edgeChild[e] != curr_state.lastSourceSinkNode) {
                        if(!edgeToDp[e]->globalSourceSink) {
                            node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                            node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                        }

                        edgeToDp[e]->globalSourceSink |= true;
                    } else {
                        node vN = e->source(), uN = e->target();
                        if((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount) {
                            if(!edgeToDp[e]->globalSourceSink) {
                                node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                                node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                            }

                            edgeToDp[e]->globalSourceSink |= true;
                        }
                    }
                }
            } else {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;
                    node vN = e->source(), uN = e->target();
                    if((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount) {
                        if(!edgeToDp[e]->globalSourceSink) {
                            node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                            node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                        }

                        edgeToDp[e]->globalSourceSink |= true;
                    }
                    
                }
            }


            std::cout << "Computing ingoing leakage.. " << std::endl;
            // computing leakage
            if(curr_state.outgoingLeakageCount >= 2) {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;

                    if(!edgeToDp[e]->hasLeakage) {
                        node_dp[edgeChild[e]].outgoingLeakageCount++;
                        node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                    }

                    edgeToDp[e]->hasLeakage |= true;
                }
            } else if(curr_state.outgoingLeakageCount == 1) {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;

                    if(edgeChild[e] != curr_state.lastLeakageNode) {
                        if(!edgeToDp[e]->hasLeakage) {
                            node_dp[edgeChild[e]].outgoingLeakageCount++;
                            node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                        }
                        edgeToDp[e]->hasLeakage |= true;
                    } else {
                        node vN = e->source(), uN = e->target();
                        if((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount) {
                            if(!edgeToDp[e]->hasLeakage) {
                                node_dp[edgeChild[e]].outgoingLeakageCount++;
                                node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                            }
                            edgeToDp[e]->hasLeakage |= true;
                        }
                    }
                }
            } else {
                for(edge e : newGraph.edges) {
                    if(!isVirtual[e]) continue;

                    node vN = e->source(), uN = e->target();
                    if((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount) {
                        if(!edgeToDp[e]->hasLeakage) {
                            node_dp[edgeChild[e]].outgoingLeakageCount++;
                            node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                        }
                        edgeToDp[e]->hasLeakage |= true;
                    }
                }
            }


            std::cout << "Computing local degrees of poles.." << std::endl;
            // updating local degrees of poles of states going into A
            for(edge e:newGraph.edges) {
                if(!isVirtual[e]) continue;
                node vN = e->source();
                node uN = e->target();
                // std::cout << vN << " -> " << uN << std::endl;

                EdgeDPState *BA = edgeToDp[e];
                EdgeDPState *AB = edgeToDpR[e];

                // std::cout << BA->s << ", " << BA->t << std::endl;
                // std::cout << AB->s << ", " << AB->t << std::endl;
                

                BA->localInS = localInDeg[mapGlobalToNew(BA->s)] - AB->localInS; 
                BA->localInT = localInDeg[mapGlobalToNew(BA->t)] - AB->localInT; 
                
                BA->localOutS = localOutDeg[mapGlobalToNew(BA->s)] - AB->localOutS; 
                BA->localOutT = localOutDeg[mapGlobalToNew(BA->t)] - AB->localOutT; 
            }
        }


        
        void tryBubble(const EdgeDPState &curr,
               const EdgeDPState &back,
               bool swap
        ) {
            // node S = swap ? curr.t : curr.s;
            // node T = swap ? curr.s : curr.t;

            // int localOutS = swap ? curr.localOutT : curr.localOutS;
            // int localOutT = swap ? curr.localOutS : curr.localOutT;
            
            // int localInS = swap ? curr.localInT : curr.localInS;
            // int localInT  = swap ? curr.localInS  : curr.localInT;

            // if (back.s == curr.s && back.t == curr.t) {
            //     localOutS += swap ? back.directTS : back.directST;
            //     localInT  += swap ? back.directST : back.directTS;
            //     if ( (swap ? back.directST : back.directTS) ) return;
            // }
            // else if (back.s == curr.t && back.t == curr.s) {
            //     localOutS += swap ? back.directST : back.directTS;
            //     localInT  += swap ? back.directTS : back.directST;
            //     if ((swap ? back.directTS : back.directST)) return;
            // }


            // if (curr.acyclic &&
            //     !curr.globalSourceSink &&
            //     !curr.hasLeakage &&
            //     ctx().outDeg[S] == localOutS &&
            //     ctx().outDeg[T] == localOutT &&
            //     ctx().inDeg[T] == localInT  &&
            //     ctx().inDeg[S] == localInS  &&
            //     !ctx().isEntry[S] &&
            //     !ctx().isExit[T])
            // {
            //     addSuperbubble(S, T);
            // }

            node S = swap ? curr.t : curr.s;
            node T = swap ? curr.s : curr.t;

            
            /* take the counts from the current direction … */
            int outS = swap ? curr.localOutT : curr.localOutS;
            int outT = swap ? curr.localOutS : curr.localOutT;
            int inS  = swap ? curr.localInT  : curr.localInS;
            int inT  = swap ? curr.localInS  : curr.localInT;

            // std::cout << ctx().node2name[S] << " -> " << ctx().node2name[T] << std::endl;

            
            /* … and add the complementary counts from the opposite state */
            // if (back.s == curr.s && back.t == curr.t) {
            //     /* same orientation */
            //     outS += swap ? back.localOutT : back.localOutS;
            //     inS  += swap ? back.localInT  : back.localInS;
            //     outT += swap ? back.localOutS : back.localOutT;
            //     inT  += swap ? back.localInS  : back.localInT;
            // }
            // else if (back.s == curr.t && back.t == curr.s) {
            //     /* opposite orientation: swap S ↔ T when adding */
            //     outS += swap ? back.localOutS : back.localOutT;
            //     inS  += swap ? back.localInS  : back.localInT;
            //     outT += swap ? back.localOutT : back.localOutS;
            //     inT  += swap ? back.localInT  : back.localInS;
            // }

            bool backGood = true;

            if (back.s == curr.s && back.t == curr.t) {
                backGood &= (!back.directTS);
            } else if (back.s == curr.t && back.t == curr.s) {
                backGood &= (!back.directST);
            }


            // std::cout << outS << " " << inS << "   " << outT << " " << inT << std::endl;

            /* combine boolean properties from *both* directions */
            bool acyclic = curr.acyclic;
            bool noLeakage = !curr.hasLeakage;
            bool noGSource = !curr.globalSourceSink;

            // std::cout << (acyclic) << ", " << (noLeakage) << ", " << (noGSource) << std::endl;

            /* ---- original test with merged numbers -------------------------------- */
            if (acyclic &&
                noGSource &&
                noLeakage &&
                backGood &&
                outS > 0 &&
                inT > 0 &&
                ctx().outDeg[S] == outS &&
                // ctx().inDeg [S] == 0  &&
                // ctx().outDeg[T] == 0 &&
                ctx().inDeg [T] == inT  &&
                !ctx().isEntry[S] &&
                !ctx().isExit [T])
            {
                addSuperbubble(S, T);
            }

        }



        void collectSuperbubbles(const BlockData &blk, EdgeArray<EdgeDP> &edge_dp) {
            const Graph &T = blk.spqr->tree();
            // std::cout << "Checking superbubbles.." << std::endl;
            for(edge e : T.edges) {
                const EdgeDPState &down = edge_dp[e].down;
                const EdgeDPState &up   = edge_dp[e].up;

                tryBubble(down, up, false);
                tryBubble(down, up, true);

                tryBubble(up, down, false);
                tryBubble(up, down, true);
            }
        }

    }





    void solveSPQR(const BlockData &blk, const CcData &cc) {
        // using namespace SPQRsolve;
        TIME_BLOCK("SPQR solving");
        auto T = blk.spqr->tree();

        GraphIO::drawGraph(T, "spqrTree");

        EdgeArray<SPQRsolve::EdgeDP> dp(T);
        NodeArray<SPQRsolve::NodeDPState> node_dp(T);


        ogdf::NodeArray<ogdf::node> parent(T, nullptr);
        std::vector<ogdf::node> nodeOrder;
        std::vector<ogdf::edge> edgeOrder;

        SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

        // std::cout << "SPQR tree edge order:\n";
        // for(auto p:edgeOrder) {
        //     std::cout << "Edge: " << p->source() << " -> " << p->target() << '\n';
        // }


        // for(ogdf::node u : T.nodes) {
        //     GraphIO::drawGraph(blk.spqr->skeleton(u).getGraph(), "skeletonOf" + to_string(u->index()));
        // }


        for(auto e:edgeOrder) {
            // printAllStates(dp, cc, blk, T);
            SPQRsolve::processEdge(e, dp, node_dp, cc, blk);
            // SPQRsolve::printAllStates(dp, node_dp, T);
        }
        

        for(auto v:nodeOrder) {
            SPQRsolve::processNode(v, dp, node_dp, cc, blk);
            // SPQRsolve::printAllStates(dp, node_dp, T);
            // break;
        }



        // return;
        // std::cout << "SPQR tree node order:\n";
        // for(auto n:nodeOrder) {
        //     std::cout << "Node: " << n << '\n';
        // }

        // std::stack<ogdf::node> S;  parent[root] = root;  S.push(root);


        // SPQRsolve::printAllStates(dp, node_dp, T);

        logger::info("Computed all dp states..");

        SPQRsolve::collectSuperbubbles(blk, dp);

    }


    void processBlock(BlockData &blk, const CcData &cc, size_t ccId, size_t  blkId) {
        std::cout << "Component #" << ccId<< "  Block #" << blkId << "  |V|=" << blk.Gblk->numberOfNodes() << "  |E|=" << blk.Gblk->numberOfEdges() << '\n';


        // check if whole block is a superbubble
        // checkWholeBlockSuperbubble(blk, cc);
        checkBlockByCutVertices(blk, cc);


        if(blk.spqr) {
            // run normal SPQR solution
            solveSPQR(blk, cc);
        }
    }



    void runOnAllBlocks(std::vector<CcData> &comps) {
        for (size_t cid = 0; cid < comps.size(); ++cid) {
            CcData &cc = comps[cid];

            for (size_t bid = 0; bid < cc.blocks.size(); ++bid) {
                processBlock(cc.blocks[bid], cc, cid, bid);
            }
        }
    }




    void findMiniSuperbubbles() {
        TIME_BLOCK("Finding mini-superbubbles");

        auto& C = ctx();


        logger::info("Finding mini-superbubbles..");

        for(auto &e:C.G.edges) {
            auto a = e->source();
            auto b = e->target();

            if(a->outdeg() == 1 && b->indeg() == 1) {
                bool ok=true;
                for(auto &w:b->adjEntries) {
                    auto e = w->theEdge();
                    auto src = e->source();
                    auto tgt = e->target();
                    if(src == b && tgt == a) {
                        ok = false;
                        break;
                    }
                }

                if(ok) {
                    addSuperbubble(a, b);
                } 
            }
        }
        logger::info("Checked for mini-superbubbles");


    }

    void solve() {                
        TIME_BLOCK("Finding superbubbles");
        auto& C = ctx();

        findMiniSuperbubbles();

        std::vector<CcData> comps = buildDecomposition(C.G);
        std::cout << "Found " << comps.size() << " connected components.\n";

        printBlockEdges(comps);
        runOnAllBlocks(comps);





    }
}





int main(int argc, char** argv) {
    TIME_BLOCK("Starting graph reading...");
    // auto& C = ctx();
    logger::init();


    // Graph G;
    // C.G = &G;


    readArgs(argc, argv);
    GraphIO::readGraph();
    GraphIO::drawGraph(ctx().G, "input_graph");



    solver::solve();


    // output all superbubbles

    std::cout << "Superbubbles found:\n";
    if(true)
    std::sort(ctx().superbubbles.begin(), ctx().superbubbles.end(), [&](std::pair<node, node> &a, std::pair<node, node> &b) {
        if(std::stoi(ctx().node2name[a.first]) < std::stoi(ctx().node2name[b.first])) return true;
        else if(std::stoi(ctx().node2name[a.first]) == std::stoi(ctx().node2name[b.first])) {
            return std::stoi(ctx().node2name[a.second]) < std::stoi(ctx().node2name[b.second]);
        }
        return false;
    });

    
    GraphIO::writeSuperbubbles();

    // for(auto &sb:ctx().superbubbles) {
    //     std::cout << ctx().node2name[sb.first] << " " << ctx().node2name[sb.second] << '\n';
    // }

    // std::cout << "total count: " << ctx().superbubbles.size() << std::endl;
    
    // std::cout << ctx().outDeg[ctx().name2node["153"]] << std::endl;

    return 0;
}