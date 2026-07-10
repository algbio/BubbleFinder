    // Public load and save functions for .spqr and .spqri.

    static IndexedSpqrTree load_text(const std::string& path)
    {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) throw std::runtime_error("cannot open SPQR text file: " + path);
        IndexedSpqrTree idx;
        idx.parse_text_stream(in, path);
        idx.build_indices();
        return idx;
    }

    static IndexedSpqrTree load_text_compressed(const std::string& path)
    {
        const std::string cmd = detail::decompressor_command(path);
        if (cmd.empty()) {
            throw std::runtime_error("no decompressor configured for: " + path);
        }

        FILE* pipe = ::popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("failed to run decompressor for: " + path);
        }

        IndexedSpqrTree idx;
        char* raw = nullptr;
        std::size_t cap = 0;
        std::uint64_t line_no = 0;
        while (true) {
            const ssize_t n = ::getline(&raw, &cap, pipe);
            if (n < 0) break;
            ++line_no;
            std::string_view line(raw, static_cast<std::size_t>(n));
            idx.parse_text_line(line, path, line_no);
        }
        std::free(raw);

        const int status = ::pclose(pipe);
        if (status != 0) {
            throw std::runtime_error("decompressor failed for: " + path);
        }

        idx.build_indices();
        return idx;
    }

    static IndexedSpqrTree load_binary(const std::string& path,
                                       const LoadOptions& options = LoadOptions{})
    {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) throw std::runtime_error("cannot open binary SPQR index: " + path);
        IndexedSpqrTree idx;
        idx.read_binary(in, path, options);
        return idx;
    }

    static IndexedSpqrTree load_binary_compressed(
        const std::string& path,
        const LoadOptions& options = LoadOptions{})
    {
#if defined(__unix__) || defined(__APPLE__)
        const std::string cmd = detail::decompressor_command(path);
        if (cmd.empty()) {
            throw std::runtime_error("no decompressor configured for: " + path);
        }

        const char* tmpdir_raw = std::getenv("TMPDIR");
        std::string tmp_path =
            (tmpdir_raw != nullptr && *tmpdir_raw != '\0')
                ? tmpdir_raw
                : "/tmp";
        if (!tmp_path.empty() && tmp_path.back() != '/')
            tmp_path.push_back('/');
        tmp_path += "bf_spqri_binary_XXXXXX";
        std::vector<char> tmp_template(tmp_path.begin(), tmp_path.end());
        tmp_template.push_back('\0');

        const int fd = ::mkstemp(tmp_template.data());
        if (fd < 0) {
            throw std::runtime_error(
                "failed to create temporary decompressed .spqri file for: " + path);
        }
        tmp_path.assign(tmp_template.data());
        struct TempFileGuard {
            std::string path;
            ~TempFileGuard()
            {
                if (!path.empty()) std::remove(path.c_str());
            }
        } guard{tmp_path};

        FILE* out = ::fdopen(fd, "wb");
        if (out == nullptr) {
            ::close(fd);
            throw std::runtime_error(
                "failed to open temporary decompressed .spqri file for: " + path);
        }
        FILE* pipe = ::popen(cmd.c_str(), "r");
        if (pipe == nullptr) {
            ::fclose(out);
            throw std::runtime_error("failed to run decompressor for: " + path);
        }

        std::vector<char> buffer(1u << 20);
        bool io_error = false;
        while (true) {
            const std::size_t n =
                std::fread(buffer.data(), 1, buffer.size(), pipe);
            if (n != 0) {
                if (std::fwrite(buffer.data(), 1, n, out) != n) {
                    io_error = true;
                    break;
                }
            }
            if (n < buffer.size()) {
                if (std::ferror(pipe)) io_error = true;
                break;
            }
        }
        const int close_status = std::fclose(out);
        const int pipe_status = ::pclose(pipe);
        if (io_error || close_status != 0) {
            throw std::runtime_error(
                "failed while writing temporary decompressed .spqri file for: " + path);
        }
        if (pipe_status != 0) {
            throw std::runtime_error("decompressor failed for: " + path);
        }

        return load_binary(tmp_path, options);
#else
        (void)options;
        throw std::runtime_error(
            "compressed .spqri binary input is not supported on this platform: " +
            path);
#endif
    }

    static IndexedSpqrTree load(const std::string& path,
                                const LoadOptions& options = LoadOptions{})
    {
        const bool compressed = detail::ends_with(path, ".gz") ||
                                detail::ends_with(path, ".bgz") ||
                                detail::ends_with(path, ".bz2") ||
                                detail::ends_with(path, ".xz");
        if (compressed) {
            if (detail::ends_with(path, ".spqri.gz") ||
                detail::ends_with(path, ".spqri.bgz") ||
                detail::ends_with(path, ".spqri.bz2") ||
                detail::ends_with(path, ".spqri.xz")) {
                return load_binary_compressed(path, options);
            }
            IndexedSpqrTree idx = load_text_compressed(path);
            idx.apply_eager_load_plan_indexes(idx.build_section_load_plan(options));
            return idx;
        }
        std::ifstream probe(path, std::ios::in | std::ios::binary);
        if (!probe) throw std::runtime_error("cannot open SPQR input: " + path);
        char magic[8] = {};
        probe.read(magic, sizeof(magic));
        if (probe.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
            std::memcmp(magic, binary_magic, sizeof(magic)) == 0) {
            return load_binary(path, options);
        }
        IndexedSpqrTree idx = load_text(path);
        idx.apply_eager_load_plan_indexes(idx.build_section_load_plan(options));
        return idx;
    }

    static constexpr std::uint32_t current_binary_version = 34u;

    static bool is_current_binary_cache(const std::string& path)
    {
        const bool compressed = detail::ends_with(path, ".gz") ||
                                detail::ends_with(path, ".bgz") ||
                                detail::ends_with(path, ".bz2") ||
                                detail::ends_with(path, ".xz");
        if (compressed) return false;

        try {
            std::ifstream in(path, std::ios::in | std::ios::binary);
            if (!in) return false;
            char magic[8] = {};
            in.read(magic, sizeof(magic));
            if (!in || std::memcmp(magic, binary_magic, sizeof(magic)) != 0) {
                return false;
            }
            const std::uint32_t version =
                detail::read_scalar<std::uint32_t>(in, "version");
            return version == current_binary_version;
        } catch (const std::exception&) {
            return false;
        }
    }

    void save_binary(const std::string& path) const
    {
        std::vector<char> buffer(4ull * 1024ull * 1024ull);
        std::filebuf file;
        file.pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        if (!file.open(path, std::ios::out | std::ios::binary)) {
            throw std::runtime_error("cannot open binary SPQR index for writing: " + path);
        }
        std::ostream out(&file);
        write_binary(out);
        if (!out) throw std::runtime_error("failed while writing binary SPQR index: " + path);
        if (file.close() == nullptr) {
            throw std::runtime_error("failed while closing binary SPQR index: " + path);
        }
    }
