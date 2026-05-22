#include "io/gbz_parser.hpp"

#include <gbwtgraph/algorithms.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nid_t = handlegraph::nid_t;

struct SegmentRange {
    gbwt::size_type segment_id = 0;
    nid_t start = 0;
    nid_t limit = 0;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [-t threads] graph.gbz\n";
}

bool parse_positive_int(const char* text, int& out) {
    if (text == nullptr || *text == '\0') return false;
    long value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10 + (*p - '0');
        if (value > std::numeric_limits<int>::max()) return false;
    }
    if (value <= 0) return false;
    out = static_cast<int>(value);
    return true;
}

std::vector<SegmentRange> collect_segments(const gbwtgraph::GBWTGraph& graph) {
    std::vector<SegmentRange> result;
    if (!graph.has_segment_names()) return result;

    auto iter = graph.node_to_segment.one_begin();
    while (iter != graph.node_to_segment.one_end()) {
        const gbwt::size_type segment_id = iter->first;
        const nid_t start = iter->second;
        ++iter;
        const nid_t limit = iter->second;
        if (graph.has_node(start)) {
            result.push_back({segment_id, start, limit});
        }
    }
    return result;
}

std::string segment_name(const gbwtgraph::GBWTGraph& graph, gbwt::size_type segment_id) {
    if (!graph.has_segment_names() || segment_id >= graph.segments.size()) {
        return "";
    }
    std::string_view view = graph.segments.view(segment_id);
    return std::string(view.data(), view.size());
}

} // namespace

int main(int argc, char** argv) {
    int threads = 1;
    std::string input;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "-t" || arg == "--threads") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], threads)) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            continue;
        }
        if (!input.empty()) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        input = std::move(arg);
    }

    if (input.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        gbwtgraph::GBZ gbz = GBZParser::load_topology_file(input, threads);
        const gbwtgraph::GBWTGraph& graph = gbz.graph;
        const auto components = gbwtgraph::weakly_connected_components(graph);
        const auto segments = collect_segments(graph);

        std::cout
            << "component\tnodes\tmin_node\tmax_node\tcontiguous_nodes"
            << "\tdisjoint_node_interval\tnext_interval_gap"
            << "\tsegments\tmin_segment\tmax_segment\tcontiguous_segments"
            << "\tfirst_segment_name\tlast_segment_name"
            << "\tbwt_comp_begin\tbwt_comp_end\tbwt_byte_begin\tbwt_byte_end\n";

        size_t segment_cursor = 0;
        for (size_t cid = 0; cid < components.size(); ++cid) {
            const auto& component = components[cid];
            if (component.empty()) continue;

            const nid_t min_node = component.front();
            const nid_t max_node = component.back();
            const bool contiguous_nodes =
                static_cast<uint64_t>(max_node - min_node + 1) == component.size();
            const bool disjoint_node_interval =
                (cid + 1 >= components.size()) || components[cid + 1].empty() ||
                max_node < components[cid + 1].front();
            const uint64_t next_interval_gap =
                (cid + 1 < components.size() && !components[cid + 1].empty() &&
                 max_node < components[cid + 1].front())
                    ? static_cast<uint64_t>(components[cid + 1].front() - max_node - 1)
                    : 0;

            while (segment_cursor < segments.size() &&
                   segments[segment_cursor].limit <= min_node) {
                ++segment_cursor;
            }

            size_t segment_count = 0;
            gbwt::size_type min_segment = std::numeric_limits<gbwt::size_type>::max();
            gbwt::size_type max_segment = 0;
            for (size_t si = segment_cursor; si < segments.size(); ++si) {
                const SegmentRange& segment = segments[si];
                if (segment.start > max_node) break;
                if (segment.limit <= min_node) continue;
                ++segment_count;
                min_segment = std::min(min_segment, segment.segment_id);
                max_segment = std::max(max_segment, segment.segment_id);
            }

            const bool has_segments = segment_count > 0;
            const bool contiguous_segments =
                has_segments &&
                (max_segment - min_segment + 1 == static_cast<gbwt::size_type>(segment_count));

            gbwt::size_type bwt_comp_begin = 0;
            gbwt::size_type bwt_comp_end = 0;
            gbwt::size_type bwt_byte_begin = 0;
            gbwt::size_type bwt_byte_end = 0;
            if (disjoint_node_interval && !gbz.index.empty()) {
                const gbwt::node_type first_node = gbwt::Node::encode(min_node, false);
                const gbwt::node_type last_node = gbwt::Node::encode(max_node, true);
                bwt_comp_begin = gbz.index.toComp(first_node);
                bwt_comp_end = gbz.index.toComp(last_node) + 1;
                bwt_byte_begin = gbz.index.bwt.getRange(bwt_comp_begin).first;
                bwt_byte_end = gbz.index.bwt.getRange(bwt_comp_end - 1).second;
            }

            const std::string first_name = has_segments ? segment_name(graph, min_segment) : "";
            const std::string last_name = has_segments ? segment_name(graph, max_segment) : "";

            std::cout
                << cid << '\t'
                << component.size() << '\t'
                << min_node << '\t'
                << max_node << '\t'
                << (contiguous_nodes ? 1 : 0) << '\t'
                << (disjoint_node_interval ? 1 : 0) << '\t'
                << next_interval_gap << '\t'
                << segment_count << '\t'
                << (has_segments ? min_segment : 0) << '\t'
                << (has_segments ? max_segment : 0) << '\t'
                << (contiguous_segments ? 1 : 0) << '\t'
                << first_name << '\t'
                << last_name << '\t'
                << bwt_comp_begin << '\t'
                << bwt_comp_end << '\t'
                << bwt_byte_begin << '\t'
                << bwt_byte_end << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "gbz_component_stats: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
