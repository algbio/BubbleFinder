#include "clsd_interface.hpp"

#include <unordered_map>
#include <set>
#include <string>
#include <utility>
#include <cstdlib> 

#include <iostream>

std::vector<std::pair<int,int>>
compute_superbubbles_from_edges(
    const std::vector<std::pair<int,int>>& edges,
    std::vector<supertree*>* out_trees 

) {
    std::set<int> node_set;
    for (const auto& e : edges) {
        node_set.insert(e.first);
        node_set.insert(e.second);
    }

    std::unordered_map<int, unsigned long> id2index;
    id2index.reserve(node_set.size());

    unsigned long idx = 0;
    for (int nid : node_set) {
        id2index[nid] = idx++;
    }

    std::unordered_map<std::string, long unsigned int*> str2int;
    str2int.reserve(node_set.size());

    for (int nid : node_set) {
        std::string sid = std::to_string(nid);
        long unsigned int* ele = new long unsigned int[3];
        ele[0] = id2index[nid];  
        ele[1] = 0;              
        ele[2] = 0;              
        str2int[sid] = ele;
    }

    for (const auto& e : edges) {
        int u = e.first;
        int v = e.second;

        auto it_u = id2index.find(u);
        auto it_v = id2index.find(v);
        if (it_u == id2index.end() || it_v == id2index.end()) {
            continue;
        }

        std::string su = std::to_string(u);
        std::string sv = std::to_string(v);

        str2int[su][1]++; 
        str2int[sv][2]++; 
    
    }

    std::vector<Vertex> vertices(node_set.size());

    for (const auto& p : str2int) {
        const std::string& sid = p.first;
        long unsigned int* ele = p.second; 
        std::pair<std::string, long unsigned int*> arg(sid, ele);
        vertices[ele[0]].init(arg);
    }

    for (const auto& e : edges) {
        int u = e.first;
        int v = e.second;

        auto it_u = id2index.find(u);
        auto it_v = id2index.find(v);
        if (it_u == id2index.end() || it_v == id2index.end()) {
            continue;
        }

        unsigned long iu = it_u->second;
        unsigned long iv = it_v->second;

        Vertex* from = &vertices[iu];
        Vertex* to   = &vertices[iv];

        from->addSuc(to);
        to->addPre(from);
    }

    for (Vertex& v : vertices) {
        v.resetCounter();
    }

    Config conf;
    conf.setVertices(static_cast<long unsigned int>(vertices.size()));
    conf.setEdges(static_cast<long unsigned int>(edges.size()));
    conf.setMultiedges(0);
    conf.trees = (out_trees != nullptr);

    conf.startClock();

    for (Vertex& v : vertices) {
        if (v.isSource()) {
            Vertex* start;
            create_postorder(&v, &start);
            conf.addOrder(start, &v, false);
            detect(start, &v, conf);
            finish(start, &v, false);
        }
    }

    for (Vertex& v : vertices) {
        if (v.stat == Vertex::CLEAN) {
            cycle_search(&v, conf);
        }
    }

    conf.endClock();


    
    std::vector<std::pair<int,int>> result;
    
    const auto& bubs = conf.getSuperbubbles();
    result.reserve(bubs.size());
    
    for (const auto& sb : bubs) {
        Vertex* entrance = sb.entrance;
        Vertex* exit     = sb.exit;
        
        int ent_id = std::stoi(entrance->getID());
        int ex_id  = std::stoi(exit->getID());
        
        result.emplace_back(ent_id, ex_id);
    }

    if(out_trees != nullptr) {
        *out_trees = conf.getTrees();
        // conf.writeTree();
        // std::cout << out_trees->size() << " supertrees computed.\n";
    }
    
    for (auto& p : str2int) {
        delete[] p.second;
    }

    return result;
}
