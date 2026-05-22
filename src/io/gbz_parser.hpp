#pragma once

#include "gfa_parser.hpp"   // BiGraph, BiLink

#include <gbwtgraph/gbz.h>
#include <gbwtgraph/gbwtgraph.h>
#include <sdsl/simple_sds.hpp>

#include <algorithm>
#include <fstream>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <ios>
#ifdef _OPENMP
#include <omp.h>
#endif

class GBZParser {
public:
    static gbwtgraph::GBZ load_topology_file(const std::string& path, int threads = 1) {
        gbwtgraph::GBZ gbz;
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open GBZ file: " + path);
        if (std::getenv("BF_GBZ_FULL_LOAD") != nullptr) {
            gbz.simple_sds_load(in);
        } else {
            load_gbz_topology_only(gbz, in, threads);
        }
        return gbz;
    }

    static BiGraph parse_file(const std::string& path, int threads = 1) {
        gbwtgraph::GBZ gbz = load_topology_file(path, threads);

        auto& graph = gbz.graph;
        BiGraph bg;

        if (graph.has_segment_names()) {
            build_from_segments(graph, bg, threads);
        } else {
            build_from_nodes(graph, bg, threads);
        }

        return bg;
    }

private:

    static void skip_stream_bytes(std::istream& in, size_t bytes) {
        if (bytes == 0) return;
        const std::streampos pos = in.tellg();
        if (pos != std::streampos(-1)) {
            in.seekg(static_cast<std::streamoff>(bytes), std::ios_base::cur);
            if (!in) {
                throw sdsl::simple_sds::InvalidData("GBZ topology load: failed to seek over data");
            }
            return;
        }
        in.ignore(static_cast<std::streamsize>(bytes));
        if (!in) {
            throw sdsl::simple_sds::InvalidData("GBZ topology load: failed to skip data");
        }
    }

    static void skip_simple_sds_data(std::istream& in, size_t bytes) {
        const size_t padded_bytes =
            sdsl::simple_sds::data_size(bytes) *
            sizeof(sdsl::simple_sds::element_type);
        skip_stream_bytes(in, padded_bytes);
    }

    static void skip_simple_sds_option_seekable(std::istream& in) {
        const size_t size = sdsl::simple_sds::load_value<size_t>(in);
        skip_stream_bytes(in, size * sizeof(sdsl::simple_sds::element_type));
    }

    template<typename Item>
    static void skip_simple_sds_vector(std::istream& in) {
        static_assert(sizeof(Item) == 1 || sizeof(Item) % sizeof(sdsl::simple_sds::element_type) == 0,
                      "The size of an item must be 1 byte or a multiple of the Simple-SDS element size");
        const size_t n = sdsl::simple_sds::load_value<size_t>(in);
        skip_simple_sds_data(in, n * sizeof(Item));
    }

    static size_t skip_zstd_even_string_array(std::istream& in) {
        sdsl::sd_vector<> index;
        index.simple_sds_load(in);
        (void)sdsl::simple_sds::load_value<size_t>(in);
        skip_simple_sds_vector<char>(in);
        return 2 * index.ones();
    }

    static void determine_real_nodes(gbwtgraph::GBWTGraph& graph,
                                     const gbwt::GBWT& index) {
        graph.header.nodes = 0;
        if (index.empty()) {
            graph.real_nodes = sdsl::bit_vector();
            return;
        }

        const gbwt::node_type first = index.firstNode();
        const size_t potential_nodes = index.sigma() - first;
        graph.real_nodes = sdsl::bit_vector(potential_nodes / 2, 0);
        for (gbwt::node_type node = first; node < index.sigma(); node += 2) {
            if (!index.empty(node)) {
                graph.real_nodes[(node - first) / 2] = 1;
                graph.header.nodes++;
            }
        }
    }

    static void load_gbz_graph_topology_only(gbwtgraph::GBWTGraph& graph,
                                             const gbwt::GBWT& index,
                                             std::istream& in,
                                             int threads) {
        graph.set_gbwt_address(index);

        gbwtgraph::GBWTGraph::Header h =
            sdsl::simple_sds::load_value<gbwtgraph::GBWTGraph::Header>(in);
        h.check();
        const bool use_zstd = (h.version >= gbwtgraph::GBWTGraph::Header::ZSTD_VERSION);
        const bool simple_sds = h.get(gbwtgraph::GBWTGraph::Header::FLAG_SIMPLE_SDS);
        h.unset(gbwtgraph::GBWTGraph::Header::FLAG_SIMPLE_SDS);
        h.set_version();
        graph.header = h;

        if (simple_sds) {
            if (threads > 1) {
                std::exception_ptr real_nodes_error;
                std::thread real_nodes_thread([&]() {
                    try {
                        determine_real_nodes(graph, index);
                    } catch (...) {
                        real_nodes_error = std::current_exception();
                    }
                });
                auto join_real_nodes = [&]() {
                    if (real_nodes_thread.joinable()) real_nodes_thread.join();
                    if (real_nodes_error) std::rethrow_exception(real_nodes_error);
                };

                try {
                    if (use_zstd) {
                        const size_t sequence_count = skip_zstd_even_string_array(in);
                        graph.sequences.index = sdsl::int_vector<>(sequence_count + 1, 0, 1);
                        graph.sequences.strings.clear();
                    } else {
                        graph.sequences.simple_sds_load_duplicate(in, gbwtgraph::reverse_complement);
                    }
                    graph.segments.simple_sds_load(in);
                    graph.node_to_segment.simple_sds_load(in);
                } catch (...) {
                    if (real_nodes_thread.joinable()) real_nodes_thread.join();
                    throw;
                }
                join_real_nodes();
            } else {
                if (use_zstd) {
                    const size_t sequence_count = skip_zstd_even_string_array(in);
                    graph.sequences.index = sdsl::int_vector<>(sequence_count + 1, 0, 1);
                    graph.sequences.strings.clear();
                } else {
                    graph.sequences.simple_sds_load_duplicate(in, gbwtgraph::reverse_complement);
                }
                determine_real_nodes(graph, index);
                graph.segments.simple_sds_load(in);
                graph.node_to_segment.simple_sds_load(in);
            }
            if (graph.header.get(gbwtgraph::GBWTGraph::Header::FLAG_TRANSLATION) !=
                (graph.segments.size() > 0)) {
                throw sdsl::simple_sds::InvalidData(
                    "GBZ topology load: invalid translation flag in GBWTGraph header");
            }
        } else {
            graph.sequences.load(in);
            graph.real_nodes.load(in);
            if (graph.header.get(gbwtgraph::GBWTGraph::Header::FLAG_TRANSLATION)) {
                graph.segments.load(in);
                graph.node_to_segment.load(in);
            }
        }
    }

    static void load_gbwt_topology_only(gbwt::GBWT& index, std::istream& in) {
        const std::streampos start = in.tellg();
        gbwt::GBWTHeader h = sdsl::simple_sds::load_value<gbwt::GBWTHeader>(in);
        h.check();
        const bool simple_sds = h.get(gbwt::GBWTHeader::FLAG_SIMPLE_SDS);
        const bool has_tags = h.version >= 5;

        if (!simple_sds) {
            if (start == std::streampos(-1)) {
                throw sdsl::simple_sds::InvalidData(
                    "GBZ topology load: non-simple-sds GBWT cannot be rewound");
            }
            in.clear();
            in.seekg(start);
            index.load(in);
            return;
        }

        h.unset(gbwt::GBWTHeader::FLAG_SIMPLE_SDS);
        h.setVersion();
        index.header = h;

        if (has_tags) {
            index.tags.simple_sds_load(in);
        } else {
            index.tags.clear();
        }

        index.bwt.simple_sds_load(in);
        if (index.bwt.size() != index.effective()) {
            throw sdsl::simple_sds::InvalidData(
                "GBZ topology load: GBWT BWT record count / alphabet size mismatch");
        }

        if (!index.empty()) {
            index.endmarker_record = gbwt::DecompressedRecord(index.record(gbwt::ENDMARKER));
        }

        skip_simple_sds_option_seekable(in);
        index.da_samples = gbwt::DASamples();

        skip_simple_sds_option_seekable(in);
        index.clearMetadata();
    }

    static void load_gbz_topology_only(gbwtgraph::GBZ& gbz, std::istream& in, int threads) {
        gbz.header = sdsl::simple_sds::load_value<gbwtgraph::GBZ::Header>(in);
        gbz.header.check();
        gbz.header.set_version();

        gbz.tags.simple_sds_load(in);
        if (std::getenv("BF_GBZ_FULL_GBWT") != nullptr) {
            gbz.index.simple_sds_load(in);
        } else {
            load_gbwt_topology_only(gbz.index, in);
        }
        load_gbz_graph_topology_only(gbz.graph, gbz.index, in, threads);
    }

    static void set_worker_count(int threads) {
#ifdef _OPENMP
        if (threads > 1) omp_set_num_threads(threads);
#else
        (void)threads;
#endif
    }

    static bool parse_canonical_u64(std::string_view s, uint64_t& out) {
        if (s.empty()) return false;
        if (s.size() > 1 && s.front() == '0') return false;
        uint64_t v = 0;
        for (char c : s) {
            if (c < '0' || c > '9') return false;
            const uint64_t digit = static_cast<uint64_t>(c - '0');
            if (v > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                return false;
            }
            v = v * 10 + digit;
        }
        out = v;
        return true;
    }

    static void build_from_segments(gbwtgraph::GBWTGraph& graph, BiGraph& bg, int threads) {
        using nid_t = handlegraph::nid_t;

        nid_t min_nid = graph.min_node_id();
        nid_t max_nid = graph.max_node_id();
        size_t range = (size_t)(max_nid - min_nid + 1);

        std::vector<uint32_t> nid_to_seg(range, UINT32_MAX);
        std::vector<std::pair<nid_t, nid_t>> seg_ranges;
        uint32_t seg_count = 0;
        const bool numeric_names = (std::getenv("BF_ENABLE_NUMERIC_GBZ_NAMES") != nullptr);
        if (numeric_names) {
            bg.numeric_node_names.reserve(graph.segments.size());
            bg.numeric_node_name_valid.reserve(graph.segments.size());
        }

        if (!numeric_names) {
            graph.for_each_segment([&](const std::string& name,
                                       std::pair<nid_t, nid_t> nodes) {
                uint32_t id = seg_count++;
                bg.node_names.push_back(name);
                seg_ranges.push_back(nodes);
                for (nid_t n = nodes.first; n < nodes.second; n++) {
                    nid_to_seg[(size_t)(n - min_nid)] = id;
                }
                return true;
            });
        } else {
            auto segment_iter = graph.node_to_segment.one_begin();
            while (segment_iter != graph.node_to_segment.one_end()) {
                const gbwt::size_type segment_id = segment_iter->first;
                const nid_t start = segment_iter->second;
                ++segment_iter;
                const nid_t limit = segment_iter->second;
                if (!graph.has_node(start)) {
                    continue;
                }

                uint32_t id = seg_count++;
                std::string_view name = graph.segments.view(segment_id);
                uint64_t parsed = 0;
                if (parse_canonical_u64(name, parsed)) {
                    bg.numeric_node_names.push_back(parsed);
                    bg.numeric_node_name_valid.push_back(1);
                } else {
                    bg.numeric_node_names.push_back(0);
                    bg.numeric_node_name_valid.push_back(0);
                    bg.string_node_names.emplace_back(
                        id, std::string(name.data(), name.size()));
                }
                std::pair<nid_t, nid_t> nodes(start, limit);
                seg_ranges.push_back(nodes);
                for (nid_t n = nodes.first; n < nodes.second; n++) {
                    nid_to_seg[(size_t)(n - min_nid)] = id;
                }
            }
        }

        const uint32_t seg_total = seg_count;
        if (threads > 1) {
            set_worker_count(threads);
        }

        if (threads <= 1) {
            bg.links.reserve(graph.get_edge_count());
            gbwt::CachedGBWT cache = graph.get_single_cache();

            auto emit = [&](const handlegraph::handle_t& from,
                            const handlegraph::handle_t& to) {
                nid_t from_nid = graph.get_id(from);
                nid_t to_nid   = graph.get_id(to);

                uint32_t src = nid_to_seg[(size_t)(from_nid - min_nid)];
                uint32_t dst = nid_to_seg[(size_t)(to_nid   - min_nid)];

                char o1 = graph.get_is_reverse(from) ? '-' : '+';
                char o2 = graph.get_is_reverse(to)   ? '-' : '+';

                bg.links.push_back({src, dst, o1, o2});
            };

            for (uint32_t sid = 0; sid < seg_total; ++sid) {
                const auto nodes = seg_ranges[sid];

                handlegraph::handle_t last = graph.get_handle(nodes.second - 1, false);
                graph.cached_follow_edges(cache, last, false, [&](const handlegraph::handle_t& next) {
                    nid_t next_id = graph.get_id(next);
                    if (next_id >= nodes.second - 1) {
                        emit(last, next);
                    }
                    return true;
                });

                handlegraph::handle_t first = graph.get_handle(nodes.first, true);
                graph.cached_follow_edges(cache, first, false, [&](const handlegraph::handle_t& next) {
                    nid_t next_id = graph.get_id(next);
                    if (next_id > nodes.first ||
                        (next_id == nodes.first && !(graph.get_is_reverse(next)))) {
                        emit(first, next);
                    }
                    return true;
                });
            }

            bg.n_nodes = seg_total;
            return;
        }

        auto append_links_for_segment = [&](uint32_t sid,
                                            std::vector<BiLink>& out_links,
                                            gbwt::CachedGBWT& cache) {
            const auto nodes = seg_ranges[sid];

            auto emit = [&](const handlegraph::handle_t& from,
                            const handlegraph::handle_t& to) {
                nid_t from_nid = graph.get_id(from);
                nid_t to_nid   = graph.get_id(to);

                uint32_t src = nid_to_seg[(size_t)(from_nid - min_nid)];
                uint32_t dst = nid_to_seg[(size_t)(to_nid   - min_nid)];

                char o1 = graph.get_is_reverse(from) ? '-' : '+';
                char o2 = graph.get_is_reverse(to)   ? '-' : '+';

                out_links.push_back({src, dst, o1, o2});
            };

            handlegraph::handle_t last = graph.get_handle(nodes.second - 1, false);
            graph.cached_follow_edges(cache, last, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id >= nodes.second - 1) {
                    emit(last, next);
                }
                return true;
            });

            handlegraph::handle_t first = graph.get_handle(nodes.first, true);
            graph.cached_follow_edges(cache, first, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id > nodes.first ||
                    (next_id == nodes.first && !(graph.get_is_reverse(next)))) {
                    emit(first, next);
                }
                return true;
            });
        };

        if (std::getenv("BF_GBZ_TWO_PASS_LINKS") == nullptr) {
            const size_t chunk_segments = 65536;
            const size_t chunk_count =
                (static_cast<size_t>(seg_total) + chunk_segments - 1) / chunk_segments;

            std::vector<std::vector<BiLink>> chunks(chunk_count);

            #pragma omp parallel for schedule(dynamic, 1) if(threads > 1 && chunk_count > 1)
            for (int64_t ci_i = 0; ci_i < static_cast<int64_t>(chunk_count); ++ci_i) {
                const size_t ci = static_cast<size_t>(ci_i);
                const uint32_t begin = static_cast<uint32_t>(ci * chunk_segments);
                const uint32_t end = static_cast<uint32_t>(
                    std::min(static_cast<size_t>(seg_total), (ci + 1) * chunk_segments));
                std::vector<BiLink>& local = chunks[ci];
                local.reserve(static_cast<size_t>(end - begin) * 2);
                gbwt::CachedGBWT cache = graph.get_single_cache();
                for (uint32_t sid = begin; sid < end; ++sid) {
                    append_links_for_segment(sid, local, cache);
                }
            }

            std::vector<size_t> chunk_offsets(chunk_count + 1, 0);
            for (size_t ci = 0; ci < chunk_count; ++ci) {
                chunk_offsets[ci + 1] = chunk_offsets[ci] + chunks[ci].size();
            }
            bg.links.resize(chunk_offsets[chunk_count]);

            #pragma omp parallel for schedule(static) if(threads > 1 && chunk_count > 1)
            for (int64_t ci_i = 0; ci_i < static_cast<int64_t>(chunk_count); ++ci_i) {
                const size_t ci = static_cast<size_t>(ci_i);
                std::copy(chunks[ci].begin(), chunks[ci].end(),
                          bg.links.begin() + static_cast<std::ptrdiff_t>(chunk_offsets[ci]));
            }

            bg.n_nodes = seg_total;
            return;
        }

        auto count_links_for_segment = [&](uint32_t sid) -> size_t {
            const auto nodes = seg_ranges[sid];
            size_t count = 0;

            handlegraph::handle_t last = graph.get_handle(nodes.second - 1, false);
            graph.follow_edges(last, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id >= nodes.second - 1) {
                    ++count;
                }
                return true;
            });

            handlegraph::handle_t first = graph.get_handle(nodes.first, true);
            graph.follow_edges(first, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id > nodes.first ||
                    (next_id == nodes.first && !(graph.get_is_reverse(next)))) {
                    ++count;
                }
                return true;
            });

            return count;
        };

        std::vector<size_t> offsets(static_cast<size_t>(seg_total) + 1, 0);
        #pragma omp parallel for schedule(dynamic, 512) if(threads > 1 && seg_total > 4096)
        for (int64_t sid = 0; sid < static_cast<int64_t>(seg_total); ++sid) {
            offsets[static_cast<size_t>(sid) + 1] =
                count_links_for_segment(static_cast<uint32_t>(sid));
        }
        for (uint32_t sid = 0; sid < seg_total; ++sid) {
            offsets[static_cast<size_t>(sid) + 1] += offsets[sid];
        }

        bg.links.resize(offsets[seg_total]);

        auto write_links_for_segment = [&](uint32_t sid) {
            const auto nodes = seg_ranges[sid];
            size_t out = offsets[sid];

            auto emit = [&](const handlegraph::handle_t& from,
                            const handlegraph::handle_t& to) {
                nid_t from_nid = graph.get_id(from);
                nid_t to_nid   = graph.get_id(to);

                uint32_t src = nid_to_seg[(size_t)(from_nid - min_nid)];
                uint32_t dst = nid_to_seg[(size_t)(to_nid   - min_nid)];

                char o1 = graph.get_is_reverse(from) ? '-' : '+';
                char o2 = graph.get_is_reverse(to)   ? '-' : '+';

                bg.links[out++] = {src, dst, o1, o2};
            };

            handlegraph::handle_t last = graph.get_handle(nodes.second - 1, false);
            graph.follow_edges(last, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id >= nodes.second - 1) {
                    emit(last, next);
                }
                return true;
            });

            handlegraph::handle_t first = graph.get_handle(nodes.first, true);
            graph.follow_edges(first, false, [&](const handlegraph::handle_t& next) {
                nid_t next_id = graph.get_id(next);
                if (next_id > nodes.first ||
                    (next_id == nodes.first && !(graph.get_is_reverse(next)))) {
                    emit(first, next);
                }
                return true;
            });
        };

        #pragma omp parallel for schedule(dynamic, 512) if(threads > 1 && seg_total > 4096)
        for (int64_t sid = 0; sid < static_cast<int64_t>(seg_total); ++sid) {
            write_links_for_segment(static_cast<uint32_t>(sid));
        }

        bg.n_nodes = seg_total;
    }

    static void build_from_nodes(gbwtgraph::GBWTGraph& graph, BiGraph& bg, int threads) {
        using nid_t = handlegraph::nid_t;

        nid_t min_nid = graph.min_node_id();
        nid_t max_nid = graph.max_node_id();
        size_t range = (size_t)(max_nid - min_nid + 1);

        if (threads > 1) set_worker_count(threads);

        std::unique_ptr<uint32_t[]> nid_to_id(new uint32_t[range]);
        uint32_t next_id = 0;

        const bool use_real_node_select =
            graph.index != nullptr && !graph.index->empty() && !graph.real_nodes.empty();
        std::unique_ptr<sdsl::bit_vector::rank_1_type> real_rank;
        std::unique_ptr<sdsl::bit_vector::select_1_type> real_select;
        if (use_real_node_select) {
            real_rank = std::make_unique<sdsl::bit_vector::rank_1_type>(&graph.real_nodes);
            real_select = std::make_unique<sdsl::bit_vector::select_1_type>(&graph.real_nodes);
        }

        bool built_mapping_from_real_nodes = false;
        if (use_real_node_select) {
            const size_t real_node_count = (*real_rank)(graph.real_nodes.size());
            if (real_node_count <= static_cast<size_t>(UINT32_MAX)) {
                bg.numeric_node_names.resize(real_node_count);
                bg.numeric_node_name_valid.assign(real_node_count, 1);

                const gbwt::node_type first_node = graph.index->firstNode();
                #pragma omp parallel for schedule(static) if(threads > 1 && real_node_count > 4096)
                for (int64_t rank_i_i = 0;
                     rank_i_i < static_cast<int64_t>(real_node_count);
                     ++rank_i_i) {
                    const size_t rank_i = static_cast<size_t>(rank_i_i) + 1;
                    const size_t rel = (*real_select)(rank_i);
                    const gbwt::node_type node =
                        first_node + static_cast<gbwt::node_type>(2 * rel);
                    const nid_t nid = static_cast<nid_t>(gbwt::Node::id(node));
                    const uint32_t id = static_cast<uint32_t>(rank_i - 1);
                    nid_to_id[static_cast<size_t>(nid - min_nid)] = id;
                    bg.numeric_node_names[id] = static_cast<uint64_t>(nid);
                }

                next_id = static_cast<uint32_t>(real_node_count);
                built_mapping_from_real_nodes = true;
            }
        }

        if (!built_mapping_from_real_nodes) {
            #pragma omp parallel for schedule(static) if(threads > 1 && range > 4096)
            for (int64_t i = 0; i < static_cast<int64_t>(range); ++i) {
                nid_to_id[static_cast<size_t>(i)] = UINT32_MAX;
            }

            graph.for_each_handle([&](const handlegraph::handle_t& h) {
                nid_t nid = graph.get_id(h);
                size_t idx = (size_t)(nid - min_nid);
                if (nid_to_id[idx] == UINT32_MAX) {
                    nid_to_id[idx] = next_id++;
                    bg.node_names.push_back(std::to_string(nid));
                }
                return true;
            });
        }

        auto append_edge = [&](const handlegraph::handle_t& from,
                               const handlegraph::handle_t& to,
                               std::vector<BiLink>& out_links) {
            nid_t from_nid = graph.get_id(from);
            nid_t to_nid   = graph.get_id(to);

            uint32_t src = nid_to_id[(size_t)(from_nid - min_nid)];
            uint32_t dst = nid_to_id[(size_t)(to_nid   - min_nid)];

            char o1 = graph.get_is_reverse(from) ? '-' : '+';
            char o2 = graph.get_is_reverse(to)   ? '-' : '+';

            out_links.push_back({src, dst, o1, o2});
        };

        auto append_gbwt_edge = [&](gbwt::node_type from,
                                    gbwt::node_type to,
                                    std::vector<BiLink>& out_links) {
            const nid_t from_nid = static_cast<nid_t>(gbwt::Node::id(from));
            const nid_t to_nid = static_cast<nid_t>(gbwt::Node::id(to));

            uint32_t src = nid_to_id[(size_t)(from_nid - min_nid)];
            uint32_t dst = nid_to_id[(size_t)(to_nid - min_nid)];

            char o1 = gbwt::Node::is_reverse(from) ? '-' : '+';
            char o2 = gbwt::Node::is_reverse(to) ? '-' : '+';

            out_links.push_back({src, dst, o1, o2});
        };

        auto append_edges_for_gbwt_node = [&](gbwt::node_type node,
                                              std::vector<BiLink>& out_links) {
            const nid_t nid = static_cast<nid_t>(gbwt::Node::id(node));

            gbwt::CompressedRecord record = graph.index->record(node);
            const gbwt::size_type outdegree = record.outdegree();
            for (gbwt::size_type outrank = 0; outrank < outdegree; ++outrank) {
                gbwt::node_type next = record.successor(outrank);
                if (next == gbwt::ENDMARKER) continue;

                const nid_t next_nid = static_cast<nid_t>(gbwt::Node::id(next));
                if (nid <= next_nid) {
                    append_gbwt_edge(node, next, out_links);
                }
            }

            const gbwt::node_type rev_node = gbwt::Node::reverse(node);
            record = graph.index->record(rev_node);
            const gbwt::size_type indegree = record.outdegree();
            for (gbwt::size_type outrank = 0; outrank < indegree; ++outrank) {
                gbwt::node_type prev = record.successor(outrank);
                if (prev == gbwt::ENDMARKER) continue;
                prev = gbwt::Node::reverse(prev);

                const nid_t prev_nid = static_cast<nid_t>(gbwt::Node::id(prev));
                if (nid < prev_nid ||
                    (nid == prev_nid && gbwt::Node::is_reverse(prev))) {
                    append_gbwt_edge(prev, node, out_links);
                }
            }
        };

        auto append_edges_for_handle_checked = [&](const handlegraph::handle_t& handle,
                                                   std::vector<BiLink>& out_links,
                                                   gbwt::CachedGBWT& cache,
                                                   nid_t job_begin,
                                                   nid_t job_end,
                                                   std::atomic<bool>* invalid_job) {
            auto emit = [&](const handlegraph::handle_t& from,
                            const handlegraph::handle_t& to) {
                if (invalid_job != nullptr) {
                    const nid_t from_nid = graph.get_id(from);
                    const nid_t to_nid = graph.get_id(to);
                    if (from_nid < job_begin || from_nid >= job_end ||
                        to_nid < job_begin || to_nid >= job_end) {
                        invalid_job->store(true, std::memory_order_relaxed);
                    }
                }
                append_edge(from, to, out_links);
            };

            graph.cached_follow_edges(cache, handle, false, [&](const handlegraph::handle_t& next) {
                if (graph.get_id(handle) <= graph.get_id(next)) {
                    emit(handle, next);
                }
                return true;
            });

            graph.cached_follow_edges(cache, handle, true, [&](const handlegraph::handle_t& prev) {
                if (graph.get_id(handle) < graph.get_id(prev) ||
                    (graph.get_id(handle) == graph.get_id(prev) && graph.get_is_reverse(prev))) {
                    emit(prev, handle);
                }
                return true;
            });
        };

        if (threads <= 1) {
            bg.links.reserve(graph.get_edge_count());
            graph.for_each_edge([&](const handlegraph::edge_t& edge) {
                append_edge(edge.first, edge.second, bg.links);
                return true;
            });
        } else {
            auto append_edges_for_node_range = [&](nid_t begin,
                                                   nid_t end,
                                                   std::vector<BiLink>& out_links,
                                                   gbwt::CachedGBWT& cache,
                                                   std::atomic<bool>* invalid_job) {
                if (!use_real_node_select) {
                    for (nid_t nid = begin; nid < end; ++nid) {
                        if (!graph.has_node(nid)) continue;
                        append_edges_for_handle_checked(graph.get_handle(nid, false),
                                                        out_links, cache,
                                                        begin, end, invalid_job);
                    }
                    return;
                }

                const gbwt::node_type first_node = graph.index->firstNode();
                const gbwt::node_type sigma = graph.index->sigma();
                gbwt::node_type first = gbwt::Node::encode(begin, false);
                gbwt::node_type limit = gbwt::Node::encode(end, false);
                if (limit <= first_node || first >= sigma) return;
                first = std::max(first, first_node);
                limit = std::min(limit, sigma);

                const size_t rel_begin =
                    static_cast<size_t>((first - first_node) / 2);
                const size_t rel_limit =
                    static_cast<size_t>((limit - first_node) / 2);
                const size_t rank_begin = (*real_rank)(rel_begin);
                const size_t rank_end = (*real_rank)(rel_limit);
                for (size_t rank_i = rank_begin + 1; rank_i <= rank_end; ++rank_i) {
                    const size_t rel = (*real_select)(rank_i);
                    const gbwt::node_type node =
                        first_node + static_cast<gbwt::node_type>(2 * rel);
                    if (invalid_job == nullptr) {
                        append_edges_for_gbwt_node(node, out_links);
                    } else {
                        append_edges_for_handle_checked(
                            gbwtgraph::GBWTGraph::node_to_handle(node),
                            out_links, cache, begin, end, invalid_job);
                    }
                }
            };

            auto move_chunks_to_graph = [&](std::vector<std::vector<BiLink>>& chunks) {
                std::vector<size_t> chunk_offsets(chunks.size() + 1, 0);
                for (size_t ci = 0; ci < chunks.size(); ++ci) {
                    chunk_offsets[ci + 1] = chunk_offsets[ci] + chunks[ci].size();
                }
                bg.links.resize(chunk_offsets[chunks.size()]);

                #pragma omp parallel for schedule(static) if(threads > 1 && chunks.size() > 1)
                for (int64_t ci_i = 0; ci_i < static_cast<int64_t>(chunks.size()); ++ci_i) {
                    const size_t ci = static_cast<size_t>(ci_i);
                    std::copy(chunks[ci].begin(), chunks[ci].end(),
                              bg.links.begin() + static_cast<std::ptrdiff_t>(chunk_offsets[ci]));
                }
            };

            const size_t chunk_nodes = 1 << 20;
            const size_t node_span = static_cast<size_t>(max_nid - min_nid + 1);
            const size_t chunk_count = (node_span + chunk_nodes - 1) / chunk_nodes;
            std::vector<std::vector<BiLink>> chunks(chunk_count);

            #pragma omp parallel for schedule(dynamic, 1) if(threads > 1 && chunk_count > 1)
            for (int64_t ci_i = 0; ci_i < static_cast<int64_t>(chunk_count); ++ci_i) {
                const size_t ci = static_cast<size_t>(ci_i);
                const nid_t begin = min_nid + static_cast<nid_t>(ci * chunk_nodes);
                const nid_t end = std::min<nid_t>(
                    max_nid + 1,
                    begin + static_cast<nid_t>(chunk_nodes));
                std::vector<BiLink>& local = chunks[ci];
                local.reserve(1 << 20);
                gbwt::CachedGBWT cache = graph.get_single_cache();
                append_edges_for_node_range(begin, end, local, cache, nullptr);
            }

            move_chunks_to_graph(chunks);
        }

        bg.n_nodes = next_id;
    }
};
