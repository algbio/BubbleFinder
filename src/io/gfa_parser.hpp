#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
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

inline const char* next_tab(const char* p, const char* end) {
    while (p < end && *p != '\t' && *p != '\n' && *p != '\r') ++p;
    return p;
}
inline const char* skip_line(const char* p, const char* end) {
    while (p < end && *p != '\n') ++p;
    return (p < end) ? p + 1 : p;
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
        return gz ? parse_gz(path, threads) : parse_mmap(path);
    }

private:
    static BiGraph parse_mmap(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open " + path);
        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); throw std::runtime_error("fstat failed"); }
        size_t sz = (size_t)st.st_size;
        if (sz == 0) { close(fd); return {}; }
        void* m = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
        madvise(m, sz, MADV_SEQUENTIAL);

        gfa_detail::ParseState state;
        state.feed((const char*)m, sz);

        munmap(m, sz);
        close(fd);
        return state.finish();
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