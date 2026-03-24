#pragma once

#include "gfa_parser.hpp"   // BiGraph, BiLink

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/gbwtgraph.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

class GBZParser {
public:
    static BiGraph parse_file(const std::string& path) {
        gbwtgraph::GBZ gbz;
        {
            std::ifstream in(path, std::ios::binary);
            if (!in) throw std::runtime_error("Cannot open GBZ file: " + path);
            gbz.simple_sds_load(in);
        }

        auto& graph = gbz.graph;
        BiGraph bg;

        if (graph.has_segment_names()) {
            build_from_segments(graph, bg);
        } else {
            build_from_nodes(graph, bg);
        }

        return bg;
    }

private:

    static void build_from_segments(gbwtgraph::GBWTGraph& graph, BiGraph& bg) {
        using nid_t = handlegraph::nid_t;

        nid_t min_nid = graph.min_node_id();
        nid_t max_nid = graph.max_node_id();
        size_t range = (size_t)(max_nid - min_nid + 1);

        std::vector<uint32_t> nid_to_seg(range, UINT32_MAX);
        uint32_t seg_count = 0;

        graph.for_each_segment([&](const std::string& name,
                                   std::pair<nid_t, nid_t> nodes) {
            uint32_t id = seg_count++;
            bg.node_names.push_back(name);
            for (nid_t n = nodes.first; n < nodes.second; n++) {
                nid_to_seg[(size_t)(n - min_nid)] = id;
            }
            return true;
        });

        bg.links.reserve(graph.get_edge_count());

        graph.for_each_link([&](const handlegraph::edge_t& edge,
                                const std::string& /*from*/,
                                const std::string& /*to*/) {
            nid_t from_nid = graph.get_id(edge.first);
            nid_t to_nid   = graph.get_id(edge.second);

            uint32_t src = nid_to_seg[(size_t)(from_nid - min_nid)];
            uint32_t dst = nid_to_seg[(size_t)(to_nid   - min_nid)];

            char o1 = graph.get_is_reverse(edge.first)  ? '-' : '+';
            char o2 = graph.get_is_reverse(edge.second) ? '-' : '+';

            bg.links.push_back({src, dst, o1, o2});
            return true;
        });

        bg.n_nodes = seg_count;
    }

    static void build_from_nodes(gbwtgraph::GBWTGraph& graph, BiGraph& bg) {
        using nid_t = handlegraph::nid_t;

        nid_t min_nid = graph.min_node_id();
        nid_t max_nid = graph.max_node_id();
        size_t range = (size_t)(max_nid - min_nid + 1);

        std::vector<uint32_t> nid_to_id(range, UINT32_MAX);
        uint32_t next_id = 0;

        graph.for_each_handle([&](const handlegraph::handle_t& h) {
            nid_t nid = graph.get_id(h);
            size_t idx = (size_t)(nid - min_nid);
            if (nid_to_id[idx] == UINT32_MAX) {
                nid_to_id[idx] = next_id++;
                bg.node_names.push_back(std::to_string(nid));
            }
            return true;
        });

        bg.links.reserve(graph.get_edge_count());

        graph.for_each_edge([&](const handlegraph::edge_t& edge) {
            nid_t from_nid = graph.get_id(edge.first);
            nid_t to_nid   = graph.get_id(edge.second);

            uint32_t src = nid_to_id[(size_t)(from_nid - min_nid)];
            uint32_t dst = nid_to_id[(size_t)(to_nid   - min_nid)];

            char o1 = graph.get_is_reverse(edge.first)  ? '-' : '+';
            char o2 = graph.get_is_reverse(edge.second) ? '-' : '+';

            bg.links.push_back({src, dst, o1, o2});
            return true;
        });

        bg.n_nodes = next_id;
    }
};