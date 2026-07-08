    // Read and write the .spqri header, sections and final table.

    static std::uint64_t tellp_u64(std::ostream& out, const char* what)
    {
        const std::streampos pos = out.tellp();
        if (pos == std::streampos(-1)) {
            throw std::runtime_error(std::string("cannot determine output position while writing ") + what);
        }
        return static_cast<std::uint64_t>(pos);
    }

    static void seekp_u64(std::ostream& out, std::uint64_t pos, const char* what)
    {
        out.seekp(static_cast<std::streamoff>(pos), std::ios::beg);
        if (!out) {
            throw std::runtime_error(std::string("cannot seek output while writing ") + what);
        }
    }

    static void seekg_u64(std::istream& in, std::uint64_t pos, const char* what)
    {
        in.seekg(static_cast<std::streamoff>(pos), std::ios::beg);
        if (!in) {
            throw std::runtime_error(std::string("cannot seek input while reading ") + what);
        }
    }

    static std::uint64_t tellg_u64(std::istream& in, const char* what)
    {
        const std::streampos pos = in.tellg();
        if (pos == std::streampos(-1)) {
            throw std::runtime_error(
                std::string("cannot determine input position while reading ") + what);
        }
        return static_cast<std::uint64_t>(pos);
    }

    static void drain_stream(std::istream& in,
                             const char* section,
                             const std::string& path,
                             const char* detail = nullptr)
    {
        char buffer[4096];
        while (in.read(buffer, sizeof(buffer))) {}
        if (in.eof()) return;
        std::string message = "failed while draining ";
        message += section;
        if (detail != nullptr) {
            message += ' ';
            message += detail;
        }
        throw std::runtime_error(message + " in " + path);
    }

    static SectionCodec selected_section_codec()
    {
        const char* codec = std::getenv("BF_SPQRI_SECTION_CODEC");
        if (codec != nullptr && *codec != '\0') {
            if (std::strcmp(codec, "raw") == 0 || std::strcmp(codec, "none") == 0) {
                return SectionCodec::Raw;
            }
#if BF_HAVE_ZSTD
            if (std::strcmp(codec, "zstd") == 0) {
                return SectionCodec::Zstd;
            }
            if (std::strcmp(codec, "zstd-chunked") == 0 ||
                std::strcmp(codec, "chunked") == 0) {
                return SectionCodec::ZstdChunked;
            }
#endif
            throw std::runtime_error(
                "unsupported BF_SPQRI_SECTION_CODEC; expected raw, zstd, or zstd-chunked");
        }
#if BF_HAVE_ZSTD
        return SectionCodec::ZstdChunked;
#else
        return SectionCodec::Raw;
#endif
    }

    static SectionCodec selected_columnar_section_codec(
        SectionCodec default_codec,
        bool prefer_raw)
    {
        const char* codec = std::getenv("BF_SPQRI_SECTION_CODEC");
        if (codec != nullptr && *codec != '\0') return default_codec;
        return prefer_raw ? SectionCodec::Raw : default_codec;
    }

    static int selected_zstd_level()
    {
        return static_cast<int>(
            selected_unsigned_env("BF_SPQRI_ZSTD_LEVEL", 1, 1, 19));
    }

    static int selected_zstd_workers()
    {
        return static_cast<int>(
            selected_unsigned_env("BF_SPQRI_ZSTD_WORKERS", 0, 0, 256));
    }

    static std::uint64_t selected_zstd_chunk_bytes()
    {
        return selected_unsigned_env(
            "BF_SPQRI_ZSTD_CHUNK_BYTES",
            64ull * 1024ull * 1024ull,
            1024ull,
            1ull * 1024ull * 1024ull * 1024ull);
    }

    static std::size_t selected_zstd_chunk_prefetch(std::size_t worker_cap = 16)
    {
        return static_cast<std::size_t>(selected_unsigned_env(
            "BF_SPQRI_ZSTD_CHUNK_PREFETCH", default_worker_count(worker_cap), 1, 64));
    }

    static void write_section_entry(std::ostream& out, const SectionEntryDisk& entry)
    {
        detail::write_scalar<std::uint32_t>(out, entry.id);
        detail::write_scalar<std::uint8_t>(out, entry.codec);
        detail::write_scalar<std::uint8_t>(out, 0u);
        detail::write_scalar<std::uint16_t>(out, 0u);
        detail::write_scalar<std::uint64_t>(out, entry.offset);
        detail::write_scalar<std::uint64_t>(out, entry.encoded_size);
        detail::write_scalar<std::uint64_t>(out, entry.decoded_size);
        detail::write_scalar<std::uint64_t>(out, 0u);
    }

    static SectionEntryDisk read_section_entry(std::istream& in)
    {
        SectionEntryDisk entry;
        entry.id = detail::read_scalar<std::uint32_t>(in, "section id");
        entry.codec = detail::read_scalar<std::uint8_t>(in, "section codec");
        (void)detail::read_scalar<std::uint8_t>(in, "section flags");
        (void)detail::read_scalar<std::uint16_t>(in, "section reserved");
        entry.offset = detail::read_scalar<std::uint64_t>(in, "section offset");
        entry.encoded_size = detail::read_scalar<std::uint64_t>(in, "section encoded size");
        entry.decoded_size = detail::read_scalar<std::uint64_t>(in, "section decoded size");
        (void)detail::read_scalar<std::uint64_t>(in, "section checksum");
        return entry;
    }

    static void write_column_payload_header(std::ostream& out,
                                            std::uint32_t id)
    {
        detail::write_scalar<std::uint32_t>(out, columnar_section_magic);
        detail::write_scalar<std::uint32_t>(out, 1u);
        detail::write_scalar<std::uint32_t>(out, id);
        detail::write_scalar<std::uint32_t>(out, 0u);
    }

    static void read_column_payload_header(std::istream& in,
                                           std::uint32_t expected_id,
                                           const std::string& path)
    {
        const std::uint32_t magic =
            detail::read_scalar<std::uint32_t>(in, "columnar section magic");
        if (magic != columnar_section_magic) {
            throw std::runtime_error("invalid columnar SPQR section in " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "columnar section version");
        if (version != 1u) {
            throw std::runtime_error("unsupported columnar SPQR section version in " + path);
        }
        const std::uint32_t id =
            detail::read_scalar<std::uint32_t>(in, "columnar section id");
        if (id != expected_id) {
            throw std::runtime_error("columnar SPQR section id mismatch in " + path);
        }
        (void)detail::read_scalar<std::uint32_t>(in, "columnar section flags");
    }

    static const SectionEntryDisk* find_section(
        const std::vector<SectionEntryDisk>& sections,
        std::uint32_t id)
    {
        auto it = std::find_if(
            sections.begin(), sections.end(),
            [id](const SectionEntryDisk& entry) {
                return entry.id == id;
            });
        return it == sections.end() ? nullptr : &*it;
    }

    static SectionEntryDisk require_section(
        const std::vector<SectionEntryDisk>& sections,
        std::uint32_t id,
        const std::string& path,
        const char* what)
    {
        const SectionEntryDisk* section = find_section(sections, id);
        if (section == nullptr) {
            throw std::runtime_error(
                std::string("binary SPQR section table is missing ") + what +
                " in " + path);
        }
        return *section;
    }

    template <typename Fn>
    static auto schedule_section_task(Fn&& fn)
        -> std::future<std::invoke_result_t<Fn>>
    {
        return spqri_section_pool().submit(std::forward<Fn>(fn));
    }

    template <typename WriteFn>
    static SectionEntryDisk write_section_payload_with_codec(
        std::ostream& out,
        std::uint32_t id,
        const char* what,
        SectionCodec codec,
        WriteFn&& write_payload,
        std::vector<SectionChunkDirectoryRecord>* chunk_directory = nullptr)
    {
        SectionEntryDisk section;
        section.id = id;
        section.codec = static_cast<std::uint8_t>(codec);
        section.offset = tellp_u64(out, what);
        (void)chunk_directory;

        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Raw)) {
            CountingRawOutputBuf raw_buf(out);
            std::ostream section_out(&raw_buf);
            write_payload(section_out);
            section_out.flush();
            if (!section_out) {
                throw std::runtime_error(std::string("failed while writing raw ") + what);
            }
            section.encoded_size = raw_buf.bytes();
            section.decoded_size = raw_buf.bytes();
        } else if (section.codec == static_cast<std::uint8_t>(SectionCodec::Zstd)) {
#if BF_HAVE_ZSTD
            ZstdOutputBuf zstd_buf(out, selected_zstd_level(), selected_zstd_workers());
            std::ostream section_out(&zstd_buf);
            write_payload(section_out);
            section_out.flush();
            zstd_buf.finish();
            if (!section_out) {
                throw std::runtime_error(std::string("failed while writing zstd ") + what);
            }
            section.encoded_size = zstd_buf.encoded_bytes();
            section.decoded_size = zstd_buf.decoded_bytes();
#else
            throw std::runtime_error("zstd SPQR sections requested but zstd support is unavailable");
#endif
        } else if (section.codec == static_cast<std::uint8_t>(SectionCodec::ZstdChunked)) {
#if BF_HAVE_ZSTD
            ZstdChunkedOutputBuf zstd_buf(
                out, selected_zstd_level(), selected_zstd_chunk_bytes());
            std::ostream section_out(&zstd_buf);
            write_payload(section_out);
            section_out.flush();
            zstd_buf.finish();
            if (!section_out) {
                throw std::runtime_error(
                    std::string("failed while writing chunked zstd ") + what);
            }
            section.encoded_size = zstd_buf.encoded_bytes();
            section.decoded_size = zstd_buf.decoded_bytes();
            if (chunk_directory != nullptr) {
                SectionChunkDirectoryRecord record;
                record.section_id = id;
                record.section_offset = section.offset;
                record.section_encoded_size = section.encoded_size;
                record.section_decoded_size = section.decoded_size;
                record.target_chunk_bytes = zstd_buf.target_chunk_bytes();
                record.chunks = zstd_buf.chunks();
                chunk_directory->push_back(std::move(record));
            }
#else
            throw std::runtime_error("chunked zstd SPQR sections requested but zstd support is unavailable");
#endif
        } else {
            throw std::runtime_error("unsupported SPQR section codec selected");
        }

        return section;
    }

    template <typename ReadFn>
    static void read_section_payload_from_file(
        const std::string& path,
        const SectionEntryDisk& section,
        const char* what,
        std::size_t chunk_prefetch,
        ReadFn&& read_payload)
    {
        (void)chunk_prefetch;
        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Raw)) {
            if (section.encoded_size != section.decoded_size) {
                throw std::runtime_error(
                    std::string("raw SPQR section size mismatch while reading ") +
                    what + " in " + path);
            }
            RawSectionInputBuf raw_buf(path, section.offset, section.encoded_size, what);
            std::istream section_in(&raw_buf);
            read_payload(section_in);
            if (raw_buf.consumed() != section.encoded_size) {
                throw std::runtime_error(
                    std::string("raw SPQR section size mismatch while reading ") +
                    what + " in " + path);
            }
            return;
        }

        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open binary SPQR index: " + path);
        }
        seekg_u64(in, section.offset, what);

        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Zstd)) {
#if BF_HAVE_ZSTD
            ZstdInputBuf zstd_buf(in, section.encoded_size, section.decoded_size);
            std::istream section_in(&zstd_buf);
            read_payload(section_in);
            drain_stream(section_in, "zstd SPQR section", path, what);
            if (zstd_buf.decoded_bytes() != section.decoded_size) {
                throw std::runtime_error(
                    std::string("zstd SPQR section size mismatch while reading ") +
                    what + " in " + path);
            }
#else
            throw std::runtime_error("zstd-compressed SPQR sections require zstd support: " + path);
#endif
        } else if (section.codec == static_cast<std::uint8_t>(SectionCodec::ZstdChunked)) {
#if BF_HAVE_ZSTD
            ZstdChunkedInputBuf zstd_buf(path, section, chunk_prefetch);
            std::istream section_in(&zstd_buf);
            read_payload(section_in);
            drain_stream(section_in, "chunked zstd SPQR section", path, what);
            if (zstd_buf.decoded_bytes() != section.decoded_size) {
                throw std::runtime_error(
                    std::string("chunked zstd SPQR section size mismatch while reading ") +
                    what + " in " + path);
            }
#else
            throw std::runtime_error("chunked zstd SPQR sections require zstd support: " + path);
#endif
        } else {
            throw std::runtime_error(
                std::string("unsupported SPQR section codec while reading ") +
                what + " in " + path);
        }
    }

    static std::vector<char> read_section_payload_bytes_from_file(
        const std::string& path,
        const SectionEntryDisk& section,
        const char* what,
        std::size_t chunk_prefetch)
    {
        (void)chunk_prefetch;
        if (section.decoded_size >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                std::string("decoded SPQR section is too large for memory: ") +
                what + " in " + path);
        }
        std::vector<char> bytes(static_cast<std::size_t>(section.decoded_size));

        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Raw)) {
            if (section.encoded_size != section.decoded_size) {
                throw std::runtime_error(
                    std::string("raw SPQR section size mismatch while reading ") +
                    what + " in " + path);
            }
            if (!bytes.empty()) {
                PreadFile file(path);
                file.read_exact(section.offset, bytes.data(), bytes.size(), what);
            }
            return bytes;
        }

        if (section.codec == static_cast<std::uint8_t>(SectionCodec::Zstd)) {
#if BF_HAVE_ZSTD
            if (section.encoded_size >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
                throw std::runtime_error(
                    std::string("encoded SPQR section is too large for memory: ") +
                    what + " in " + path);
            }
            std::vector<char> encoded(
                static_cast<std::size_t>(section.encoded_size));
            if (!encoded.empty()) {
                PreadFile file(path);
                file.read_exact(section.offset,
                                encoded.data(),
                                encoded.size(),
                                what);
            }
            if (!bytes.empty()) {
                const std::size_t rc =
                    ZSTD_decompressDCtx(reusable_zstd_dctx(),
                                        bytes.data(),
                                        bytes.size(),
                                        encoded.data(),
                                        encoded.size());
                if (ZSTD_isError(rc)) {
                    throw std::runtime_error(
                        std::string("failed while decompressing zstd SPQR section ") +
                        what + " in " + path + ": " + ZSTD_getErrorName(rc));
                }
                if (rc != bytes.size()) {
                    throw std::runtime_error(
                        std::string("zstd SPQR section size mismatch while reading ") +
                        what + " in " + path);
                }
            }
            return bytes;
#else
            throw std::runtime_error("zstd-compressed SPQR sections require zstd support: " + path);
#endif
        }

        if (section.codec == static_cast<std::uint8_t>(SectionCodec::ZstdChunked)) {
#if BF_HAVE_ZSTD
            const std::shared_ptr<const ZstdChunkTableData> table =
                ZstdChunkedInputBuf::load_cached_chunk_table(path, section);
            const std::shared_ptr<const PreadFile> file =
                std::make_shared<PreadFile>(path);
            std::vector<std::future<void>> futures;
            auto chunk_output = [&](const ZstdChunkEntryDisk& chunk) {
                if (chunk.decoded_offset > section.decoded_size ||
                    chunk.decoded_size >
                        section.decoded_size - chunk.decoded_offset) {
                    throw std::runtime_error(
                        "zstd chunk decoded bounds exceed section size in " + path);
                }
                return bytes.data() + static_cast<std::size_t>(chunk.decoded_offset);
            };
            if (table->chunks.size() > 1) {
                futures.reserve(table->chunks.size());
                for (const ZstdChunkEntryDisk& chunk : table->chunks) {
                    char* dst = chunk_output(chunk);
                    futures.push_back(spqri_chunk_pool().submit(
                        [file, chunk, dst]() {
                            ZstdChunkedInputBuf::decompress_chunk_into_buffer(
                                file, chunk, dst);
                        }));
                }
                for (auto& future : futures) future.get();
            } else {
                for (const ZstdChunkEntryDisk& chunk : table->chunks) {
                    char* dst = chunk_output(chunk);
                    ZstdChunkedInputBuf::decompress_chunk_into_buffer(file, chunk, dst);
                }
            }
            return bytes;
#else
            throw std::runtime_error("chunked zstd SPQR sections require zstd support: " + path);
#endif
        }

        throw std::runtime_error(
            std::string("unsupported SPQR section codec while reading ") +
            what + " in " + path);
    }

    static void write_zstd_chunk_directory_payload(
        std::ostream& out,
        const std::vector<SectionChunkDirectoryRecord>& records)
    {
        detail::write_scalar<std::uint32_t>(out, zstd_chunk_directory_magic);
        detail::write_scalar<std::uint32_t>(out, 1u);
        detail::write_scalar<std::uint64_t>(out, 0u);
        detail::write_scalar<std::uint64_t>(
            out, static_cast<std::uint64_t>(records.size()));
        for (const SectionChunkDirectoryRecord& record : records) {
            detail::write_scalar<std::uint32_t>(out, record.section_id);
            detail::write_scalar<std::uint32_t>(out, 0u);
            detail::write_scalar<std::uint64_t>(out, record.section_offset);
            detail::write_scalar<std::uint64_t>(out, record.section_encoded_size);
            detail::write_scalar<std::uint64_t>(out, record.section_decoded_size);
            detail::write_scalar<std::uint64_t>(out, record.target_chunk_bytes);
            detail::write_scalar<std::uint64_t>(
                out, static_cast<std::uint64_t>(record.chunks.size()));
            for (const ZstdChunkEntryDisk& chunk : record.chunks) {
                detail::write_scalar<std::uint64_t>(out, chunk.offset);
                detail::write_scalar<std::uint64_t>(out, chunk.encoded_size);
                detail::write_scalar<std::uint64_t>(out, chunk.decoded_offset);
                detail::write_scalar<std::uint64_t>(out, chunk.decoded_size);
            }
        }
    }

    static void preload_zstd_chunk_directory(
        const std::string& path,
        const std::vector<SectionEntryDisk>& sections)
    {
#if BF_HAVE_ZSTD
        const SectionEntryDisk* directory =
            find_section(sections, section_id_zstd_chunk_directory);
        if (directory == nullptr) return;
        read_section_payload_from_file(
            path,
            *directory,
            "zstd chunk directory section",
            1,
            [&](std::istream& in) {
                const std::uint32_t magic =
                    detail::read_scalar<std::uint32_t>(
                        in, "zstd chunk directory magic");
                if (magic != zstd_chunk_directory_magic) {
                    throw std::runtime_error(
                        "invalid zstd chunk directory in " + path);
                }
                const std::uint32_t version =
                    detail::read_scalar<std::uint32_t>(
                        in, "zstd chunk directory version");
                if (version != 1u) {
                    throw std::runtime_error(
                        "unsupported zstd chunk directory version in " + path);
                }
                (void)detail::read_scalar<std::uint64_t>(
                    in, "zstd chunk directory flags");
                const std::uint32_t record_count =
                    read_count(in, "zstd chunk directory records");
                for (std::uint32_t i = 0; i < record_count; ++i) {
                    SectionEntryDisk section;
                    section.id = detail::read_scalar<std::uint32_t>(
                        in, "zstd chunk directory section id");
                    (void)detail::read_scalar<std::uint32_t>(
                        in, "zstd chunk directory reserved");
                    section.offset = detail::read_scalar<std::uint64_t>(
                        in, "zstd chunk directory section offset");
                    section.encoded_size = detail::read_scalar<std::uint64_t>(
                        in, "zstd chunk directory section encoded size");
                    section.decoded_size = detail::read_scalar<std::uint64_t>(
                        in, "zstd chunk directory section decoded size");
                    if (section.encoded_size >
                        std::numeric_limits<std::uint64_t>::max() - section.offset) {
                        throw std::runtime_error(
                            "zstd chunk directory section bounds overflow in " + path);
                    }
                    const std::uint64_t section_end =
                        section.offset + section.encoded_size;
                    ZstdChunkTableData table;
                    table.target_chunk_bytes =
                        detail::read_scalar<std::uint64_t>(
                            in, "zstd chunk directory target chunk bytes");
                    const std::uint32_t chunk_count =
                        read_count(in, "zstd chunk directory chunks");
                    table.chunks.reserve(chunk_count);

                    std::uint64_t decoded_offset = 0;
                    for (std::uint32_t j = 0; j < chunk_count; ++j) {
                        ZstdChunkEntryDisk chunk;
                        chunk.offset = detail::read_scalar<std::uint64_t>(
                            in, "zstd chunk directory chunk offset");
                        chunk.encoded_size = detail::read_scalar<std::uint64_t>(
                            in, "zstd chunk directory chunk encoded size");
                        chunk.decoded_offset = detail::read_scalar<std::uint64_t>(
                            in, "zstd chunk directory chunk decoded offset");
                        chunk.decoded_size = detail::read_scalar<std::uint64_t>(
                            in, "zstd chunk directory chunk decoded size");
                        if (chunk.encoded_size >
                            std::numeric_limits<std::uint64_t>::max() - chunk.offset) {
                            throw std::runtime_error(
                                "zstd chunk directory chunk bounds overflow in " + path);
                        }
                        const std::uint64_t chunk_end =
                            chunk.offset + chunk.encoded_size;
                        if (chunk.offset < section.offset || chunk_end > section_end) {
                            throw std::runtime_error(
                                "zstd chunk directory chunk is outside its section in " +
                                path);
                        }
                        if (chunk.decoded_offset != decoded_offset) {
                            throw std::runtime_error(
                                "zstd chunk directory decoded offsets are not contiguous in " +
                                path);
                        }
                        if (chunk.decoded_size >
                            std::numeric_limits<std::uint64_t>::max() - decoded_offset) {
                            throw std::runtime_error(
                                "zstd chunk directory decoded total overflow in " + path);
                        }
                        decoded_offset += chunk.decoded_size;
                        table.chunks.push_back(chunk);
                    }
                    if (decoded_offset != section.decoded_size) {
                        throw std::runtime_error(
                            "zstd chunk directory decoded total mismatch in " + path);
                    }

                    const SectionEntryDisk* table_section = find_section(
                        sections, section.id);
                    if (table_section == nullptr ||
                        table_section->offset != section.offset ||
                        table_section->encoded_size != section.encoded_size ||
                        table_section->decoded_size != section.decoded_size ||
                        table_section->codec !=
                            static_cast<std::uint8_t>(SectionCodec::ZstdChunked)) {
                        throw std::runtime_error(
                            "zstd chunk directory references an unknown section in " +
                            path);
                    }
                    section.codec = table_section->codec;
                    ZstdChunkedInputBuf::cache_chunk_table(
                        path, section, std::move(table));
                }
            });
#else
        (void)path;
        (void)sections;
#endif
    }

    void write_binary(std::ostream& out) const
    {
        if (!runtime_indices_ready) {
            throw std::runtime_error("cannot write binary SPQR index before runtime indices are built");
        }
        out.write(binary_magic, 8);
        detail::write_scalar<std::uint32_t>(out, current_binary_version);
        detail::write_string(out, format_version);
        detail::write_string(out, format_url);
        detail::write_string(out, canonical_graph_profile(graph_view));
        write_columnar_sectioned_runtime_binary(out);
    }

    void read_sectioned_runtime_binary(std::istream& in,
                                       const std::string& path,
                                       const LoadOptions& options)
    {
        const std::uint32_t magic =
            detail::read_scalar<std::uint32_t>(in, "sectioned runtime magic");
        if (magic != sectioned_runtime_magic) {
            throw std::runtime_error("invalid sectioned binary SPQR runtime in " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "sectioned runtime version");
        if (version != 2u) {
            throw std::runtime_error("unsupported sectioned binary SPQR runtime version in " + path);
        }
        (void)detail::read_scalar<std::uint64_t>(in, "sectioned runtime flags");
        const std::uint64_t table_offset =
            detail::read_scalar<std::uint64_t>(in, "section table offset");

        seekg_u64(in, table_offset, "section table");
        const std::uint32_t table_magic =
            detail::read_scalar<std::uint32_t>(in, "section table magic");
        if (table_magic != section_table_magic) {
            throw std::runtime_error("invalid binary SPQR section table in " + path);
        }
        const std::uint32_t table_version =
            detail::read_scalar<std::uint32_t>(in, "section table version");
        if (table_version != 1u) {
            throw std::runtime_error("unsupported binary SPQR section table version in " + path);
        }
        const std::uint64_t section_count64 =
            detail::read_scalar<std::uint64_t>(in, "section count");
        const std::uint32_t section_count =
            graph_index::require_spqr_count(section_count64, "binary SPQR section count");
        std::vector<SectionEntryDisk> sections;
        sections.reserve(section_count);
        for (std::uint32_t i = 0; i < section_count; ++i) {
            sections.push_back(read_section_entry(in));
        }

        read_columnar_runtime_binary(path, sections, options);
    }

    void read_binary(std::istream& in,
                     const std::string& path,
                     const LoadOptions& options = LoadOptions{})
    {
        char magic[8] = {};
        in.read(magic, sizeof(magic));
        if (!in || std::memcmp(magic, binary_magic, sizeof(magic)) != 0) {
            throw std::runtime_error("not a BubbleFinder binary SPQR index: " + path);
        }
        const std::uint32_t version =
            detail::read_scalar<std::uint32_t>(in, "version");
        if (version != current_binary_version) {
            throw std::runtime_error("unsupported binary SPQR index version in " + path);
        }
        format_version = detail::read_string(in, "format version");
        format_url = detail::read_string(in, "format url");
        graph_view = canonical_graph_profile(
            detail::read_string(in, "graph view"));
        read_sectioned_runtime_binary(in, path, options);
    }

    template <typename T, std::size_t Size>
    static constexpr bool direct_record_size =
        std::is_standard_layout<T>::value &&
        std::is_trivially_copyable<T>::value && sizeof(T) == Size;

    static_assert(direct_record_size<GraphEdgeRecord, 8u>,
                  "GraphEdgeRecord must remain an eight-byte direct-readable record");
    static_assert(direct_record_size<HaplotypePathRecord, 48u>,
                  "HaplotypePathRecord must remain a compact direct-readable record");
    static_assert(direct_record_size<HaplotypeStepRecord, 8u>,
                  "HaplotypeStepRecord must remain an eight-byte direct-readable record");
    static_assert(direct_record_size<SlimBlockRecord, 32u>,
                  "SlimBlockRecord must remain a 32-byte direct-readable record");
    static_assert(direct_record_size<SlimVirtualEdgeRecord, 16u>,
                  "SlimVirtualEdgeRecord must remain a 16-byte direct-readable record");
    static_assert(direct_record_size<MacroCacheNodeRecord, 32u>,
                  "MacroCacheNodeRecord must match the SP-compress C node layout");
    static_assert(direct_record_size<MacroCacheCoreEdgeRecord, 16u>,
                  "MacroCacheCoreEdgeRecord must match the SP-compress C core-edge layout");
    static_assert(direct_record_size<MacroCacheSkeletonEdgeRecord, 24u>,
                  "MacroCacheSkeletonEdgeRecord must match the Rust skeleton-edge C layout");
    static_assert(direct_record_size<MacroCacheBlockRecord, 304u>,
                  "MacroCacheBlockRecord must remain a 304-byte direct-readable record");
