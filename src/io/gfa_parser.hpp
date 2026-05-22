#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__linux__) && !defined(F_SETPIPE_SZ)
#define F_SETPIPE_SZ 1031
#endif

struct BiLink {
    uint32_t src;
    uint32_t dst;
    char     orient_src;
    char     orient_dst;
};

struct BiGraph {
    uint32_t                 n_nodes = 0;
    std::vector<std::string> node_names;
    std::vector<uint64_t>    numeric_node_names;
    std::vector<uint8_t>     numeric_node_name_valid;
    std::vector<std::pair<uint32_t, std::string>> string_node_names;
    std::vector<BiLink>      links;
};

namespace gfa_detail {

struct StringView {
    const char* ptr;
    uint32_t    len;
};

inline uint64_t sv_hash(StringView s) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < s.len; ++i) {
        h ^= (uint8_t)s.ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct NameMap {
    struct Slot {
        uint64_t hash       = 0;
        uint32_t id         = UINT32_MAX;
        uint32_t name_off   = 0;
        uint32_t name_len   = 0;
        bool     used       = false;
    };

    std::vector<Slot> slots;
    uint32_t          mask  = 0;
    uint32_t          count = 0;
    std::vector<char> store;

    void init(uint32_t cap) {
        uint32_t sz = 1;
        while (sz < cap * 2) sz <<= 1;
        slots.resize(sz);
        mask = sz - 1;
        store.reserve(cap * 8);
    }

    std::pair<uint32_t, bool> get_or_insert(StringView sv, uint32_t next_id) {
        uint64_t h = sv_hash(sv);
        uint32_t idx = (uint32_t)(h & mask);
        while (true) {
            auto& s = slots[idx];
            if (!s.used) {
                s.hash = h; s.id = next_id;
                s.name_off = (uint32_t)store.size();
                s.name_len = sv.len;
                store.insert(store.end(), sv.ptr, sv.ptr + sv.len);
                s.used = true;
                count++;
                if (count * 2 > slots.size()) rehash();
                return {next_id, true};
            }
            if (s.hash == h && s.name_len == sv.len &&
                memcmp(&store[s.name_off], sv.ptr, sv.len) == 0)
                return {s.id, false};
            idx = (idx + 1) & mask;
        }
    }

    uint32_t find(StringView sv) const {
        if (slots.empty()) return UINT32_MAX;
        uint64_t h = sv_hash(sv);
        uint32_t idx = (uint32_t)(h & mask);
        while (true) {
            const auto& s = slots[idx];
            if (!s.used) return UINT32_MAX;
            if (s.hash == h && s.name_len == sv.len &&
                memcmp(&store[s.name_off], sv.ptr, sv.len) == 0)
                return s.id;
            idx = (idx + 1) & mask;
        }
    }

    void rehash() {
        uint32_t nc = (uint32_t)slots.size() * 2;
        std::vector<Slot> old = std::move(slots);
        slots.resize(nc); mask = nc - 1;
        for (auto& s : old) {
            if (!s.used) continue;
            uint32_t idx = (uint32_t)(s.hash & mask);
            while (slots[idx].used) idx = (idx + 1) & mask;
            slots[idx] = s;
        }
    }

    std::vector<std::string> to_names() const {
        std::vector<std::string> v(count);
        for (auto& s : slots)
            if (s.used) v[s.id] = std::string(&store[s.name_off], s.name_len);
        return v;
    }
};

struct SegmentRef {
    const char* ptr = nullptr;
    uint32_t len = 0;
};

struct ChunkRange {
    size_t begin = 0;
    size_t end = 0;
};

struct ChunkScan {
    std::vector<SegmentRef> segments;
    size_t link_count = 0;
    size_t first_link_offset = std::numeric_limits<size_t>::max();
    size_t last_segment_offset = 0;
};

struct NumericScan {
    bool ok = true;
    uint32_t n_nodes = 0;
    std::vector<uint32_t> id_map;
    std::vector<SegmentRef> name_refs;
    std::vector<size_t> link_offsets;
};

inline const char* next_tab(const char* p, const char* end) {
    while (p < end && *p != '\t' && *p != '\n' && *p != '\r') ++p;
    return p;
}
inline const char* skip_line(const char* p, const char* end) {
    while (p < end && *p != '\n') ++p;
    return (p < end) ? p + 1 : p;
}

inline bool parse_u32_token(const char*& p, const char* end, uint32_t& out) {
    if (p >= end || *p < '0' || *p > '9') return false;
    uint64_t v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        v = v * 10u + static_cast<uint64_t>(*p - '0');
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        ++p;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

inline bool ensure_numeric_name(NumericScan& scan,
                                uint32_t numeric_id,
                                const char* name,
                                uint32_t name_len) {
    constexpr uint32_t kMaxDenseNumericId = 100'000'000;
    if (numeric_id == 0) return false;
    if (numeric_id > kMaxDenseNumericId) return false;
    if (numeric_id >= scan.id_map.size()) {
        size_t next_size = std::max<size_t>(numeric_id + 1, scan.id_map.size() * 2);
        scan.id_map.resize(next_size, UINT32_MAX);
    }

    uint32_t& mapped = scan.id_map[numeric_id];
    if (mapped == UINT32_MAX) {
        mapped = scan.n_nodes++;
        scan.name_refs.push_back({name, name_len});
        return true;
    }

    const SegmentRef& ref = scan.name_refs[mapped];
    return ref.len == name_len && std::memcmp(ref.ptr, name, name_len) == 0;
}

inline NumericScan scan_numeric_gfa_in_serial_order(const char* data, size_t size) {
    NumericScan scan;
    scan.id_map.assign(1 << 20, UINT32_MAX);
    const char* p = data;
    const char* end = data + size;

    while (p < end) {
        const char* line = p;
        if (*p == 'S' && p + 1 < end && p[1] == '\t') {
            p += 2;
            const char* name = p;
            uint32_t id = 0;
            if (!parse_u32_token(p, end, id) || p >= end || *p != '\t') {
                scan.ok = false;
                return scan;
            }
            uint32_t name_len = static_cast<uint32_t>(p - name);
            if (!ensure_numeric_name(scan, id, name, name_len)) {
                scan.ok = false;
                return scan;
            }
            p = skip_line(p, end);
        } else if (*p == 'L' && p + 1 < end && p[1] == '\t') {
            scan.link_offsets.push_back(static_cast<size_t>(line - data));
            p += 2;

            const char* src_name = p;
            uint32_t src = 0;
            if (!parse_u32_token(p, end, src) || p >= end || *p != '\t') {
                scan.ok = false;
                return scan;
            }
            uint32_t src_len = static_cast<uint32_t>(p - src_name);
            if (!ensure_numeric_name(scan, src, src_name, src_len)) {
                scan.ok = false;
                return scan;
            }
            ++p;
            if (p >= end || (*p != '+' && *p != '-')) {
                scan.ok = false;
                return scan;
            }
            ++p;
            if (p >= end || *p != '\t') {
                scan.ok = false;
                return scan;
            }
            ++p;

            const char* dst_name = p;
            uint32_t dst = 0;
            if (!parse_u32_token(p, end, dst) || p >= end || *p != '\t') {
                scan.ok = false;
                return scan;
            }
            uint32_t dst_len = static_cast<uint32_t>(p - dst_name);
            if (!ensure_numeric_name(scan, dst, dst_name, dst_len)) {
                scan.ok = false;
                return scan;
            }
            ++p;
            if (p >= end || (*p != '+' && *p != '-')) {
                scan.ok = false;
                return scan;
            }
            p = skip_line(p, end);
        } else {
            p = skip_line(p, end);
        }
    }

    return scan;
}

inline bool parse_numeric_link_at(const char* base,
                                  const char* end,
                                  size_t offset,
                                  const std::vector<uint32_t>& id_map,
                                  BiLink& out) {
    const char* p = base + offset;
    if (p >= end || *p != 'L' || p + 1 >= end || p[1] != '\t') return false;
    p += 2;

    uint32_t src = 0;
    if (!parse_u32_token(p, end, src) || p >= end || *p != '\t') return false;
    ++p;
    if (p >= end || (*p != '+' && *p != '-')) return false;
    char o1 = *p++;
    if (p >= end || *p != '\t') return false;
    ++p;

    uint32_t dst = 0;
    if (!parse_u32_token(p, end, dst) || p >= end || *p != '\t') return false;
    ++p;
    if (p >= end || (*p != '+' && *p != '-')) return false;
    char o2 = *p;

    if (src >= id_map.size() || dst >= id_map.size()) return false;
    uint32_t src_id = id_map[src];
    uint32_t dst_id = id_map[dst];
    if (src_id == UINT32_MAX || dst_id == UINT32_MAX) return false;
    out = {src_id, dst_id, o1, o2};
    return true;
}

inline std::vector<ChunkRange> make_line_chunks(const char* data,
                                                size_t begin,
                                                size_t end,
                                                int threads,
                                                bool force_parallel = false) {
    size_t size = end - begin;
    size_t workers = (threads > 1) ? (size_t)threads : 1;
    if (!force_parallel) {
        workers = std::min(workers, std::max<size_t>(1, size / (16 << 20)));
    }
    if (workers <= 1) return {{begin, end}};

    std::vector<size_t> bounds(workers + 1, 0);
    bounds[0] = begin;
    bounds[workers] = end;
    for (size_t i = 1; i < workers; ++i) {
        size_t pos = begin + (size * i) / workers;
        while (pos < end && data[pos - 1] != '\n') ++pos;
        bounds[i] = pos;
    }

    std::vector<ChunkRange> ranges;
    ranges.reserve(workers);
    for (size_t i = 0; i < workers; ++i) {
        if (bounds[i] < bounds[i + 1]) ranges.push_back({bounds[i], bounds[i + 1]});
    }
    return ranges;
}

inline std::vector<ChunkRange> make_line_chunks(const char* data,
                                                size_t size,
                                                int threads,
                                                bool force_parallel = false) {
    return make_line_chunks(data, 0, size, threads, force_parallel);
}

inline ChunkScan scan_segments_and_count_links(const char* base, ChunkRange range) {
    ChunkScan out;
    const char* p = base + range.begin;
    const char* end = base + range.end;

    while (p < end) {
        const char* line = p;
        if (*p == 'S' && p + 1 < end && p[1] == '\t') {
            p += 2;
            const char* s = p;
            p = next_tab(p, end);
            uint32_t len = (uint32_t)(p - s);
            if (len > 0) {
                out.segments.push_back({s, len});
                out.last_segment_offset = (size_t)(line - base);
            }
            p = skip_line(p, end);
        } else if (*p == 'L' && p + 1 < end && p[1] == '\t') {
            p += 2;
            p = next_tab(p, end);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            p++;
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            p = next_tab(p, end);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            ++out.link_count;
            size_t off = (size_t)(line - base);
            if (off < out.first_link_offset) out.first_link_offset = off;
            p = skip_line(p, end);
        } else {
            p = skip_line(p, end);
        }
    }

    return out;
}

inline size_t fill_links_from_chunk(const char* base,
                                    ChunkRange range,
                                    const NameMap& names,
                                    BiLink* out_links,
                                    std::atomic<bool>& missing_name) {
    size_t written = 0;
    const char* p = base + range.begin;
    const char* end = base + range.end;

    while (p < end) {
        if (*p == 'L' && p + 1 < end && p[1] == '\t') {
            p += 2;
            const char* fs = p; p = next_tab(p, end);
            uint32_t fl = (uint32_t)(p - fs);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            char o1 = *p; p++;
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            const char* ts = p; p = next_tab(p, end);
            uint32_t tl = (uint32_t)(p - ts);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            char o2 = *p;

            uint32_t uid = names.find({fs, fl});
            uint32_t vid = names.find({ts, tl});
            if (uid == UINT32_MAX || vid == UINT32_MAX) {
                missing_name.store(true, std::memory_order_relaxed);
            } else {
                out_links[written] = {uid, vid, o1, o2};
            }
            ++written;
            p = skip_line(p, end);
        } else {
            p = skip_line(p, end);
        }
    }

    return written;
}

inline size_t count_links_in_chunk(const char* base,
                                   ChunkRange range,
                                   std::atomic<bool>& saw_late_segment) {
    size_t count = 0;
    const char* p = base + range.begin;
    const char* end = base + range.end;

    while (p < end) {
        if (*p == 'S' && p + 1 < end && p[1] == '\t') {
            saw_late_segment.store(true, std::memory_order_relaxed);
            p = skip_line(p, end);
        } else if (*p == 'L' && p + 1 < end && p[1] == '\t') {
            p += 2;
            p = next_tab(p, end);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            p++;
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            p = next_tab(p, end);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            ++count;
            p = skip_line(p, end);
        } else {
            p = skip_line(p, end);
        }
    }

    return count;
}

inline void parse_links_to_vector(const char* base,
                                  ChunkRange range,
                                  const NameMap& names,
                                  std::vector<BiLink>& out_links,
                                  std::atomic<bool>& missing_name,
                                  std::atomic<bool>& saw_late_segment) {
    out_links.reserve(std::max<size_t>(1024, (range.end - range.begin) / 32));
    const char* p = base + range.begin;
    const char* end = base + range.end;

    while (p < end) {
        if (*p == 'S' && p + 1 < end && p[1] == '\t') {
            saw_late_segment.store(true, std::memory_order_relaxed);
            p = skip_line(p, end);
        } else if (*p == 'L' && p + 1 < end && p[1] == '\t') {
            p += 2;
            const char* fs = p; p = next_tab(p, end);
            uint32_t fl = (uint32_t)(p - fs);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            char o1 = *p; p++;
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            const char* ts = p; p = next_tab(p, end);
            uint32_t tl = (uint32_t)(p - ts);
            if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
            p++;
            if (p >= end) { p = skip_line(p, end); continue; }
            char o2 = *p;

            uint32_t uid = names.find({fs, fl});
            uint32_t vid = names.find({ts, tl});
            if (uid == UINT32_MAX || vid == UINT32_MAX) {
                missing_name.store(true, std::memory_order_relaxed);
            } else {
                out_links.push_back({uid, vid, o1, o2});
            }
            p = skip_line(p, end);
        } else {
            p = skip_line(p, end);
        }
    }
}

struct ParseState {
    NameMap  names;
    uint32_t next_id = 0;
    std::vector<BiLink> links;

    ParseState() { names.init(1 << 20); }

    void feed(const char* data, size_t size) {
        const char* p   = data;
        const char* end = data + size;

        while (p < end) {
            if (*p == 'S' && p+1 < end && p[1] == '\t') {
                p += 2;
                const char* s = p;
                p = next_tab(p, end);
                if ((uint32_t)(p - s) > 0) {
                    auto [id, nw] = names.get_or_insert({s, (uint32_t)(p-s)}, next_id);
                    if (nw) next_id++;
                }
                p = skip_line(p, end);
            }
            else if (*p == 'L' && p+1 < end && p[1] == '\t') {
                p += 2;
                const char* fs = p; p = next_tab(p, end);
                uint32_t fl = (uint32_t)(p - fs);
                if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
                p++;
                char o1 = *p; p++;
                if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
                p++;
                const char* ts = p; p = next_tab(p, end);
                uint32_t tl = (uint32_t)(p - ts);
                if (p >= end || *p != '\t') { p = skip_line(p, end); continue; }
                p++;
                char o2 = *p;

                auto [uid, un] = names.get_or_insert({fs, fl}, next_id);
                if (un) next_id++;
                auto [vid, vn] = names.get_or_insert({ts, tl}, next_id);
                if (vn) next_id++;

                links.push_back({uid, vid, o1, o2});
                p = skip_line(p, end);
            }
            else {
                p = skip_line(p, end);
            }
        }
    }

    BiGraph finish() {
        BiGraph bg;
        bg.n_nodes    = next_id;
        bg.node_names = names.to_names();
        bg.links      = std::move(links);
        return bg;
    }
};


inline bool has_pigz() {
    static int cached = -1;
    if (cached < 0)
        cached = (system("pigz --version >/dev/null 2>&1") == 0) ? 1 : 0;
    return cached == 1;
}

inline std::string decompress_cmd(const std::string& path, int threads) {
    int decomp_threads = std::max(1, threads - 1);  // reserve 1 for parser
    if (decomp_threads > 1 && has_pigz())
        return "pigz -dc -p " + std::to_string(decomp_threads) + " '" + path + "'";
    return "gzip -dc '" + path + "'";
}


inline void enlarge_pipe(FILE* pipe, size_t target = 1 << 20) {
#ifdef F_SETPIPE_SZ
    int fd = fileno(pipe);
    if (fd >= 0)
        fcntl(fd, F_SETPIPE_SZ, (int)target);
#else
    (void)pipe; (void)target;
#endif
}

}

class GFAParser {
public:
    static BiGraph parse_file(const std::string& path, int threads = 1) {
        bool gz = (path.size() >= 3 && path.compare(path.size()-3, 3, ".gz") == 0) ||
                  (path.size() >= 4 && path.compare(path.size()-4, 4, ".bgz") == 0);
        return gz ? parse_gz(path, threads) : parse_mmap(path, threads);
    }

private:
    static BiGraph parse_buffer_serial(const char* data, size_t size) {
        gfa_detail::ParseState state;
        state.feed(data, size);
        return state.finish();
    }

    static BiGraph parse_buffer_parallel(const char* data,
                                         size_t size,
                                         int threads,
                                         bool force_parallel) {
        gfa_detail::NameMap names;
        size_t estimated_segments = std::max<size_t>(1 << 20, size / 140);
        if (estimated_segments > std::numeric_limits<uint32_t>::max()) {
            estimated_segments = std::numeric_limits<uint32_t>::max();
        }
        names.init((uint32_t)estimated_segments);

        const char* p = data;
        const char* end = data + size;
        uint32_t next_id = 0;
        size_t link_begin = size;

        while (p < end) {
            const char* line = p;
            if (*p == 'S' && p + 1 < end && p[1] == '\t') {
                p += 2;
                const char* s = p;
                p = gfa_detail::next_tab(p, end);
                uint32_t len = (uint32_t)(p - s);
                if (len > 0) {
                    auto [id, is_new] = names.get_or_insert({s, len}, next_id);
                    (void)id;
                    if (is_new) ++next_id;
                }
                p = gfa_detail::skip_line(p, end);
            } else if (*p == 'L' && p + 1 < end && p[1] == '\t') {
                link_begin = (size_t)(line - data);
                break;
            } else {
                p = gfa_detail::skip_line(p, end);
            }
        }

        if (link_begin == size) {
            BiGraph bg;
            bg.n_nodes = next_id;
            bg.node_names = names.to_names();
            return bg;
        }

        auto ranges = gfa_detail::make_line_chunks(
            data, link_begin, size, threads, force_parallel);
        if (ranges.size() <= 1) return parse_buffer_serial(data, size);

        std::atomic<bool> saw_late_segment{false};
        std::atomic<bool> missing_name{false};
        std::vector<std::vector<BiLink>> chunk_links(ranges.size());
        std::vector<std::thread> workers;
        workers.reserve(ranges.size());
        for (size_t i = 0; i < ranges.size(); ++i) {
            workers.emplace_back([&, i]() {
                gfa_detail::parse_links_to_vector(
                    data, ranges[i], names, chunk_links[i],
                    missing_name, saw_late_segment);
            });
        }
        for (auto& th : workers) th.join();

        if (saw_late_segment.load(std::memory_order_relaxed) ||
            missing_name.load(std::memory_order_relaxed)) {
            return parse_buffer_serial(data, size);
        }

        size_t link_count = 0;
        for (const auto& local : chunk_links) {
            link_count += local.size();
        }

        std::vector<BiLink> links;
        links.reserve(link_count);
        for (auto& local : chunk_links) {
            links.insert(links.end(),
                         std::make_move_iterator(local.begin()),
                         std::make_move_iterator(local.end()));
            std::vector<BiLink>().swap(local);
        }
        std::vector<std::vector<BiLink>>().swap(chunk_links);

        BiGraph bg;
        bg.n_nodes = next_id;
        bg.node_names = names.to_names();
        bg.links = std::move(links);
        return bg;
    }

    static BiGraph parse_buffer_numeric_dense(const char* data,
                                              size_t size,
                                              int threads) {
        gfa_detail::NumericScan scan = gfa_detail::scan_numeric_gfa_in_serial_order(data, size);
        if (!scan.ok) {
            std::vector<size_t>().swap(scan.link_offsets);
            std::vector<uint32_t>().swap(scan.id_map);
            std::vector<gfa_detail::SegmentRef>().swap(scan.name_refs);
            return parse_buffer_serial(data, size);
        }

        std::vector<BiLink> links(scan.link_offsets.size());
        const char* end = data + size;
        std::atomic<bool> ok{true};
        size_t workers = (threads > 1) ? static_cast<size_t>(threads) : 1;
        workers = std::min(workers, std::max<size_t>(1, links.size() / (1 << 20)));

        if (workers <= 1) {
            for (size_t i = 0; i < scan.link_offsets.size(); ++i) {
                if (!gfa_detail::parse_numeric_link_at(
                        data, end, scan.link_offsets[i], scan.id_map, links[i])) {
                    ok.store(false, std::memory_order_relaxed);
                    break;
                }
            }
        } else {
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (size_t w = 0; w < workers; ++w) {
                size_t begin = (scan.link_offsets.size() * w) / workers;
                size_t finish = (scan.link_offsets.size() * (w + 1)) / workers;
                pool.emplace_back([&, begin, finish]() {
                    for (size_t i = begin; i < finish; ++i) {
                        if (!gfa_detail::parse_numeric_link_at(
                                data, end, scan.link_offsets[i], scan.id_map, links[i])) {
                            ok.store(false, std::memory_order_relaxed);
                            break;
                        }
                    }
                });
            }
            for (auto& th : pool) th.join();
        }

        std::vector<size_t>().swap(scan.link_offsets);
        if (!ok.load(std::memory_order_relaxed)) {
            std::vector<BiLink>().swap(links);
            std::vector<uint32_t>().swap(scan.id_map);
            std::vector<gfa_detail::SegmentRef>().swap(scan.name_refs);
            return parse_buffer_serial(data, size);
        }

        BiGraph bg;
        bg.n_nodes = scan.n_nodes;
        bg.node_names.resize(scan.n_nodes);
        for (uint32_t i = 0; i < scan.n_nodes; ++i) {
            const auto& ref = scan.name_refs[i];
            bg.node_names[i] = std::string(ref.ptr, ref.len);
        }
        std::vector<uint32_t>().swap(scan.id_map);
        std::vector<gfa_detail::SegmentRef>().swap(scan.name_refs);
        bg.links = std::move(links);
        return bg;
    }

    static BiGraph parse_mmap(const std::string& path, int threads) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); throw std::runtime_error("fstat failed"); }
        size_t sz = (size_t)st.st_size;
        if (sz == 0) { close(fd); return {}; }
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
        madvise(m, sz, MADV_SEQUENTIAL);

        const char* data = (const char*)m;
        BiGraph bg;
        const char* parallel_env = std::getenv("BF_GFA_PAR_PARSE");
        const char* numeric_env = std::getenv("BF_GFA_NUMERIC_PARSE");
        bool force_serial = parallel_env && std::string(parallel_env) == "0";
        bool force_parallel = parallel_env && std::string(parallel_env) == "1";
        bool numeric_parse = numeric_env && std::string(numeric_env) == "1";
        if (!force_serial && numeric_parse) {
            bg = parse_buffer_numeric_dense(data, sz, threads);
        } else if (!force_serial && force_parallel && threads > 1) {
            bg = parse_buffer_parallel(data, sz, threads, force_parallel);
        } else {
            bg = parse_buffer_serial(data, sz);
        }

        munmap(m, sz);
        close(fd);
        return bg;
    }

    static constexpr size_t CHUNK = 4 << 20;

    static BiGraph parse_gz(const std::string& path, int threads) {
        std::string cmd = gfa_detail::decompress_cmd(path, threads);
        fprintf(stderr, "[gfa_parser] decompressor: %s\n", cmd.c_str());
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) throw std::runtime_error("Cannot decompress " + path);

        gfa_detail::enlarge_pipe(pipe, 1 << 20);

        gfa_detail::ParseState state;

        std::vector<char> buf(CHUNK + (1 << 20));
        size_t leftover = 0;

        while (true) {
            size_t space = buf.size() - leftover;
            if (space < CHUNK) {
                buf.resize(leftover + CHUNK);
                space = CHUNK;
            }

            size_t nread = fread(buf.data() + leftover, 1, CHUNK, pipe);
            size_t total = leftover + nread;

            if (total == 0) break;

            bool eof = (nread < CHUNK);

            if (eof) {
                state.feed(buf.data(), total);
                leftover = 0;
            } else {
                size_t last_nl = total;
                while (last_nl > 0 && buf[last_nl - 1] != '\n') --last_nl;

                if (last_nl > 0) {
                    state.feed(buf.data(), last_nl);
                    leftover = total - last_nl;
                    if (leftover > 0)
                        memmove(buf.data(), buf.data() + last_nl, leftover);
                } else {
                    leftover = total;
                }
            }
        }

        pclose(pipe);
        return state.finish();
    }
};
