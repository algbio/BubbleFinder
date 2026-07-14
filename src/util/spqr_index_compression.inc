    // mmap, zstd and worker code used for compressed sections.

    class TaskPool {
    public:
        explicit TaskPool(std::size_t workers)
        {
            workers = std::max<std::size_t>(1, workers);
            threads_.reserve(workers);
            for (std::size_t i = 0; i < workers; ++i) {
                threads_.emplace_back([this]() { worker_loop(); });
            }
        }

        ~TaskPool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            cv_.notify_all();
            for (std::thread& thread : threads_) {
                if (thread.joinable()) thread.join();
            }
        }

        TaskPool(const TaskPool&) = delete;
        TaskPool& operator=(const TaskPool&) = delete;

        template <typename Fn>
        auto submit(Fn&& fn) -> std::future<std::invoke_result_t<Fn>>
        {
            using Result = std::invoke_result_t<Fn>;
            auto task = std::make_shared<std::packaged_task<Result()>>(
                std::forward<Fn>(fn));
            std::future<Result> future = task->get_future();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_) {
                    throw std::runtime_error("SPQRI task pool is shutting down");
                }
                tasks_.emplace([task]() { (*task)(); });
            }
            cv_.notify_one();
            return future;
        }

    private:
        void worker_loop()
        {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() {
                        return stopping_ || !tasks_.empty();
                    });
                    if (stopping_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        }

        std::vector<std::thread> threads_;
        std::queue<std::function<void()>> tasks_;
        std::mutex mutex_;
        std::condition_variable cv_;
        bool stopping_ = false;
    };

    static std::size_t default_worker_count(std::size_t cap)
    {
        const unsigned hw = std::thread::hardware_concurrency();
        return std::max<std::size_t>(
            1, std::min<std::size_t>(cap, hw == 0 ? 1 : hw));
    }

    static std::uint64_t selected_unsigned_env(const char* name,
                                               std::uint64_t fallback,
                                               std::uint64_t min_value,
                                               std::uint64_t max_value)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') return fallback;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' ||
            parsed < min_value || parsed > max_value) {
            throw std::runtime_error(
                std::string(name) + " must be an integer in [" +
                std::to_string(min_value) + ", " +
                std::to_string(max_value) + "]");
        }
        return static_cast<std::uint64_t>(parsed);
    }

    static std::size_t selected_worker_count(const char* env_name,
                                             std::size_t fallback,
                                             std::size_t hard_max)
    {
        return static_cast<std::size_t>(
            selected_unsigned_env(env_name, fallback, 1, hard_max));
    }

    static TaskPool& spqri_section_pool()
    {
        static TaskPool pool(selected_worker_count(
            "BF_SPQRI_SECTION_WORKERS", default_worker_count(16), 256));
        return pool;
    }

    static TaskPool& spqri_chunk_pool()
    {
        static TaskPool pool(selected_worker_count(
            "BF_SPQRI_IO_WORKERS", default_worker_count(32), 512));
        return pool;
    }

    static std::size_t checked_size(std::uint64_t value, const char* what)
    {
        if (value > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(std::string(what) + " does not fit in size_t");
        }
        return static_cast<std::size_t>(value);
    }

    class MemoryInputBuf : public std::streambuf {
    public:
        MemoryInputBuf(const char* data, std::size_t size)
        {
            char* begin = size == 0 ? empty_ : const_cast<char*>(data);
            setg(begin, begin, begin + size);
        }

    private:
        char empty_[1] = {};
    };

    class PreadFile {
    public:
        explicit PreadFile(std::string path) : path_(std::move(path))
        {
#if defined(__unix__) || defined(__APPLE__)
            fd_ = ::open(path_.c_str(), O_RDONLY);
            if (fd_ < 0) {
                throw std::runtime_error(
                    "cannot open binary SPQR index: " + path_ + ": " +
                    std::strerror(errno));
            }
#endif
        }

        ~PreadFile()
        {
#if defined(__unix__) || defined(__APPLE__)
            if (fd_ >= 0) ::close(fd_);
#endif
        }

        PreadFile(const PreadFile&) = delete;
        PreadFile& operator=(const PreadFile&) = delete;

        void read_exact(std::uint64_t offset,
                        void* data,
                        std::size_t bytes,
                        const char* what) const
        {
            char* p = static_cast<char*>(data);
#if defined(__unix__) || defined(__APPLE__)
            while (bytes != 0) {
                const std::size_t chunk = std::min<std::size_t>(
                    bytes, static_cast<std::size_t>(1ull << 30));
                const ssize_t got = ::pread(
                    fd_, p, chunk, static_cast<off_t>(offset));
                if (got < 0) {
                    if (errno == EINTR) continue;
                    throw std::runtime_error(
                        std::string("failed while reading ") + what +
                        " from " + path_ + ": " + std::strerror(errno));
                }
                if (got == 0) {
                    throw std::runtime_error(
                        std::string("truncated binary SPQR index data: ") +
                        what);
                }
                p += got;
                offset += static_cast<std::uint64_t>(got);
                bytes -= static_cast<std::size_t>(got);
            }
#else
            std::ifstream in(path_, std::ios::in | std::ios::binary);
            if (!in) throw std::runtime_error("cannot open binary SPQR index: " + path_);
            in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            if (!in) {
                throw std::runtime_error(
                    std::string("cannot seek input while reading ") + what);
            }
            while (bytes != 0) {
                const std::size_t chunk = std::min<std::size_t>(
                    bytes, static_cast<std::size_t>(1ull << 30));
                in.read(p, static_cast<std::streamsize>(chunk));
                if (!in) {
                    throw std::runtime_error(
                        std::string("truncated binary SPQR index data: ") +
                        what);
                }
                p += chunk;
                bytes -= chunk;
            }
#endif
        }

    private:
        std::string path_;
#if defined(__unix__) || defined(__APPLE__)
        int fd_ = -1;
#endif
    };

    class RawSectionInputBuf : public std::streambuf {
    public:
        RawSectionInputBuf(std::string path,
                           std::uint64_t offset,
                           std::uint64_t bytes,
                           const char* what)
            : path_(std::move(path)),
              next_offset_(offset),
              remaining_(bytes),
              what_(what)
        {
            setg(single_, single_, single_);
            try_mmap(offset, bytes);
            if (mapped_data_ == nullptr) {
                file_ = std::make_shared<PreadFile>(path_);
            }
        }

        ~RawSectionInputBuf() override
        {
#if defined(__unix__) || defined(__APPLE__)
            if (mapped_base_ != nullptr && mapped_base_ != MAP_FAILED) {
                ::munmap(mapped_base_, mapped_size_);
            }
            if (fd_ >= 0) {
                ::close(fd_);
            }
#endif
        }

        RawSectionInputBuf(const RawSectionInputBuf&) = delete;
        RawSectionInputBuf& operator=(const RawSectionInputBuf&) = delete;

        std::uint64_t consumed() const
        {
            if (mapped_data_ != nullptr) {
                return static_cast<std::uint64_t>(gptr() - eback());
            }
            return consumed_;
        }

    protected:
        std::streamsize xsgetn(char* s, std::streamsize n) override
        {
            if (n <= 0) return 0;
            if (mapped_data_ != nullptr) {
                return std::streambuf::xsgetn(s, n);
            }

            const std::uint64_t want = std::min<std::uint64_t>(
                static_cast<std::uint64_t>(n), remaining_);
            if (want == 0) return 0;
            const std::size_t bytes = checked_size(want, what_);
            file_->read_exact(next_offset_, s, bytes, what_);
            next_offset_ += want;
            remaining_ -= want;
            consumed_ += want;
            return static_cast<std::streamsize>(bytes);
        }

        int underflow() override
        {
            if (mapped_data_ != nullptr) {
                if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
                return traits_type::eof();
            }
            if (remaining_ == 0) return traits_type::eof();
            file_->read_exact(next_offset_, single_, 1, what_);
            ++next_offset_;
            --remaining_;
            ++consumed_;
            setg(single_, single_, single_ + 1);
            return traits_type::to_int_type(*gptr());
        }

    private:
        void try_mmap(std::uint64_t offset, std::uint64_t bytes)
        {
#if defined(__unix__) || defined(__APPLE__)
            if (bytes == 0 ||
                bytes > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
                return;
            }

            const long page_size_long = ::sysconf(_SC_PAGESIZE);
            if (page_size_long <= 0) return;
            const std::uint64_t page_size =
                static_cast<std::uint64_t>(page_size_long);
            const std::uint64_t map_offset = offset - (offset % page_size);
            const std::uint64_t delta = offset - map_offset;
            if (bytes > std::numeric_limits<std::uint64_t>::max() - delta) return;
            const std::uint64_t map_size64 = bytes + delta;
            if (map_size64 >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
                return;
            }

            fd_ = ::open(path_.c_str(), O_RDONLY);
            if (fd_ < 0) return;
            mapped_size_ = static_cast<std::size_t>(map_size64);
            mapped_base_ = ::mmap(nullptr,
                                  mapped_size_,
                                  PROT_READ,
                                  MAP_PRIVATE,
                                  fd_,
                                  static_cast<off_t>(map_offset));
            if (mapped_base_ == MAP_FAILED) {
                mapped_base_ = nullptr;
                mapped_size_ = 0;
                ::close(fd_);
                fd_ = -1;
                return;
            }
            mapped_data_ = static_cast<char*>(mapped_base_) + delta;
            setg(mapped_data_,
                 mapped_data_,
                 mapped_data_ + static_cast<std::size_t>(bytes));
#else
            (void)offset;
            (void)bytes;
#endif
        }

        std::string path_;
        std::shared_ptr<PreadFile> file_;
        std::uint64_t next_offset_ = 0;
        std::uint64_t remaining_ = 0;
        std::uint64_t consumed_ = 0;
        const char* what_ = "raw SPQR section";
        char single_[1] = {};
        char* mapped_data_ = nullptr;
#if defined(__unix__) || defined(__APPLE__)
        int fd_ = -1;
        void* mapped_base_ = nullptr;
        std::size_t mapped_size_ = 0;
#endif
    };

    class CountingRawOutputBuf : public std::streambuf {
    public:
        explicit CountingRawOutputBuf(std::ostream& sink) : sink_(sink) {}

        std::uint64_t bytes() const { return bytes_; }

    protected:
        std::streamsize xsputn(const char* s, std::streamsize n) override
        {
            sink_.write(s, n);
            if (!sink_) return 0;
            bytes_ += static_cast<std::uint64_t>(n);
            return n;
        }

        int overflow(int ch) override
        {
            if (ch == traits_type::eof()) return traits_type::not_eof(ch);
            const char c = static_cast<char>(ch);
            sink_.write(&c, 1);
            if (!sink_) return traits_type::eof();
            ++bytes_;
            return ch;
        }

        int sync() override
        {
            sink_.flush();
            return sink_ ? 0 : -1;
        }

    private:
        std::ostream& sink_;
        std::uint64_t bytes_ = 0;
    };

    static std::size_t selected_zstd_chunk_write_pipeline()
    {
        return selected_worker_count(
            "BF_SPQRI_ZSTD_CHUNK_WRITE_PIPELINE",
            default_worker_count(4),
            64);
    }

#if BF_HAVE_ZSTD
    struct ThreadLocalZstdDCtx {
        ThreadLocalZstdDCtx()
        {
            ptr = ZSTD_createDCtx();
            if (ptr == nullptr) {
                throw std::runtime_error("failed to create reusable zstd decompressor");
            }
        }

        ~ThreadLocalZstdDCtx()
        {
            if (ptr != nullptr) ZSTD_freeDCtx(ptr);
        }

        ZSTD_DCtx* ptr = nullptr;
    };

    struct ThreadLocalZstdCCtx {
        ThreadLocalZstdCCtx()
        {
            ptr = ZSTD_createCCtx();
            if (ptr == nullptr) {
                throw std::runtime_error("failed to create reusable zstd compressor");
            }
        }

        ~ThreadLocalZstdCCtx()
        {
            if (ptr != nullptr) ZSTD_freeCCtx(ptr);
        }

        ZSTD_CCtx* ptr = nullptr;
    };

    static ZSTD_DCtx* reusable_zstd_dctx()
    {
        thread_local ThreadLocalZstdDCtx dctx;
        return dctx.ptr;
    }

    static ZSTD_CCtx* reusable_zstd_cctx()
    {
        thread_local ThreadLocalZstdCCtx cctx;
        return cctx.ptr;
    }

    class ZstdOutputBuf : public std::streambuf {
    public:
        ZstdOutputBuf(std::ostream& sink, int level, int workers)
            : sink_(sink), outbuf_(1u << 20)
        {
            stream_ = ZSTD_createCStream();
            if (stream_ == nullptr) {
                throw std::runtime_error("failed to create zstd compression stream");
            }
            const std::size_t rc_reset =
                ZSTD_CCtx_reset(stream_, ZSTD_reset_session_and_parameters);
            if (ZSTD_isError(rc_reset)) {
                throw std::runtime_error(
                    std::string("failed to reset zstd compression stream: ") +
                    ZSTD_getErrorName(rc_reset));
            }
            const std::size_t rc_level =
                ZSTD_CCtx_setParameter(stream_, ZSTD_c_compressionLevel, level);
            if (ZSTD_isError(rc_level)) {
                throw std::runtime_error(
                    std::string("failed to set zstd compression level: ") +
                    ZSTD_getErrorName(rc_level));
            }
            if (workers > 0) {
                const std::size_t rc_workers =
                    ZSTD_CCtx_setParameter(stream_, ZSTD_c_nbWorkers, workers);
                if (ZSTD_isError(rc_workers)) {
                    throw std::runtime_error(
                        std::string("failed to enable zstd worker threads: ") +
                        ZSTD_getErrorName(rc_workers));
                }
            }
        }

        ~ZstdOutputBuf() override
        {
            if (stream_ != nullptr) ZSTD_freeCStream(stream_);
        }

        std::uint64_t encoded_bytes() const { return encoded_bytes_; }
        std::uint64_t decoded_bytes() const { return decoded_bytes_; }

        void finish()
        {
            if (finished_) return;
            std::size_t remaining = 0;
            do {
                ZSTD_outBuffer output{outbuf_.data(), outbuf_.size(), 0};
                ZSTD_inBuffer input{nullptr, 0, 0};
                remaining = ZSTD_compressStream2(
                    stream_, &output, &input, ZSTD_e_end);
                if (ZSTD_isError(remaining)) {
                    throw std::runtime_error(
                        std::string("failed while finishing zstd section: ") +
                        ZSTD_getErrorName(remaining));
                }
                write_output(output);
            } while (remaining != 0);
            finished_ = true;
        }

    protected:
        std::streamsize xsputn(const char* s, std::streamsize n) override
        {
            compress_bytes(s, static_cast<std::size_t>(n));
            return n;
        }

        int overflow(int ch) override
        {
            if (ch == traits_type::eof()) return traits_type::not_eof(ch);
            const char c = static_cast<char>(ch);
            compress_bytes(&c, 1);
            return ch;
        }

        int sync() override
        {
            sink_.flush();
            return sink_ ? 0 : -1;
        }

    private:
        void write_output(const ZSTD_outBuffer& output)
        {
            if (output.pos == 0) return;
            sink_.write(static_cast<const char*>(output.dst),
                        static_cast<std::streamsize>(output.pos));
            if (!sink_) throw std::runtime_error("failed while writing zstd section");
            encoded_bytes_ += static_cast<std::uint64_t>(output.pos);
        }

        void compress_bytes(const char* data, std::size_t size)
        {
            if (finished_) {
                throw std::runtime_error("attempted to write after finishing zstd section");
            }
            decoded_bytes_ += static_cast<std::uint64_t>(size);
            ZSTD_inBuffer input{data, size, 0};
            while (input.pos < input.size) {
                ZSTD_outBuffer output{outbuf_.data(), outbuf_.size(), 0};
                const std::size_t rc = ZSTD_compressStream2(
                    stream_, &output, &input, ZSTD_e_continue);
                if (ZSTD_isError(rc)) {
                    throw std::runtime_error(
                        std::string("failed while compressing zstd section: ") +
                        ZSTD_getErrorName(rc));
                }
                write_output(output);
            }
        }

        std::ostream& sink_;
        ZSTD_CStream* stream_ = nullptr;
        std::vector<char> outbuf_;
        std::uint64_t encoded_bytes_ = 0;
        std::uint64_t decoded_bytes_ = 0;
        bool finished_ = false;
    };

    class ZstdInputBuf : public std::streambuf {
    public:
        ZstdInputBuf(std::istream& source,
                     std::uint64_t encoded_size,
                     std::uint64_t decoded_size)
            : source_(source),
              remaining_encoded_(encoded_size),
              expected_decoded_(decoded_size),
              inbuf_(1u << 20),
              outbuf_(1u << 20)
        {
            stream_ = ZSTD_createDStream();
            if (stream_ == nullptr) {
                throw std::runtime_error("failed to create zstd decompression stream");
            }
            const std::size_t rc = ZSTD_initDStream(stream_);
            if (ZSTD_isError(rc)) {
                throw std::runtime_error(
                    std::string("failed to initialize zstd decompression stream: ") +
                    ZSTD_getErrorName(rc));
            }
            setg(outbuf_.data(), outbuf_.data(), outbuf_.data());
        }

        ~ZstdInputBuf() override
        {
            if (stream_ != nullptr) ZSTD_freeDStream(stream_);
        }

        std::uint64_t decoded_bytes() const { return decoded_bytes_; }

    protected:
        std::streamsize xsgetn(char* s, std::streamsize n) override
        {
            if (n <= 0) return 0;

            std::streamsize copied = 0;
            while (copied < n) {
                if (gptr() < egptr()) {
                    const std::streamsize buffered =
                        std::min<std::streamsize>(egptr() - gptr(), n - copied);
                    std::memcpy(s + copied, gptr(), static_cast<std::size_t>(buffered));
                    gbump(static_cast<int>(buffered));
                    copied += buffered;
                    continue;
                }

                setg(outbuf_.data(), outbuf_.data(), outbuf_.data());
                const std::size_t produced = decompress_into(
                    s + copied, static_cast<std::size_t>(n - copied));
                if (produced == 0) break;
                copied += static_cast<std::streamsize>(produced);
            }
            return copied;
        }

        int underflow() override
        {
            if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
            const std::size_t produced =
                decompress_into(outbuf_.data(), outbuf_.size());
            if (produced == 0) return traits_type::eof();
            setg(outbuf_.data(), outbuf_.data(), outbuf_.data() + produced);
            return traits_type::to_int_type(*gptr());
        }

    private:
        bool refill_input()
        {
            if (input_.pos < input_.size) return true;
            if (remaining_encoded_ == 0) {
                if (decoded_bytes_ != expected_decoded_) {
                    throw std::runtime_error("zstd section decoded size mismatch");
                }
                return false;
            }

            const std::size_t want = static_cast<std::size_t>(
                std::min<std::uint64_t>(inbuf_.size(), remaining_encoded_));
            source_.read(inbuf_.data(), static_cast<std::streamsize>(want));
            const std::streamsize got = source_.gcount();
            if (got <= 0) {
                throw std::runtime_error("truncated zstd section");
            }
            remaining_encoded_ -= static_cast<std::uint64_t>(got);
            input_ = ZSTD_inBuffer{
                inbuf_.data(), static_cast<std::size_t>(got), 0};
            return true;
        }

        std::size_t decompress_into(char* dst, std::size_t capacity)
        {
            if (capacity == 0) return 0;
            while (true) {
                if (!refill_input()) return 0;

                ZSTD_outBuffer output{dst, capacity, 0};
                const std::size_t rc = ZSTD_decompressStream(stream_, &output, &input_);
                if (ZSTD_isError(rc)) {
                    throw std::runtime_error(
                        std::string("failed while decompressing zstd section: ") +
                        ZSTD_getErrorName(rc));
                }
                if (output.pos != 0) {
                    decoded_bytes_ += static_cast<std::uint64_t>(output.pos);
                    if (decoded_bytes_ > expected_decoded_) {
                        throw std::runtime_error("zstd section decoded beyond declared size");
                    }
                    return output.pos;
                }
                if (rc == 0 && remaining_encoded_ == 0 && input_.pos == input_.size) {
                    if (decoded_bytes_ != expected_decoded_) {
                        throw std::runtime_error("zstd section decoded size mismatch");
                    }
                    return 0;
                }
            }
        }

        std::istream& source_;
        std::uint64_t remaining_encoded_ = 0;
        std::uint64_t expected_decoded_ = 0;
        ZSTD_DStream* stream_ = nullptr;
        std::vector<char> inbuf_;
        std::vector<char> outbuf_;
        ZSTD_inBuffer input_{nullptr, 0, 0};
        std::uint64_t decoded_bytes_ = 0;
    };

    // Chunk payloads are written first; their lookup table comes last:
    //
    //   [small header][chunk 0][chunk 1] ... [chunk table]

    class ZstdChunkedOutputBuf : public std::streambuf {
    public:
        ZstdChunkedOutputBuf(std::ostream& sink,
                             int level,
                             std::uint64_t target_chunk_bytes)
            : sink_(sink),
              level_(level),
              target_chunk_bytes_(checked_size(target_chunk_bytes,
                                               "zstd chunk target size")),
              max_pending_compressed_(selected_zstd_chunk_write_pipeline())
        {
            if (target_chunk_bytes_ == 0) {
                throw std::runtime_error("zstd chunk target size must be non-zero");
            }

            section_start_ = tellp_u64(sink_, "zstd chunked section");
            detail::write_scalar<std::uint32_t>(sink_, zstd_chunked_section_magic);
            detail::write_scalar<std::uint32_t>(sink_, 1u);
            detail::write_scalar<std::uint64_t>(sink_, 0u);
            decoded_size_field_ = tellp_u64(sink_, "zstd chunked decoded size field");
            detail::write_scalar<std::uint64_t>(sink_, 0u);
            detail::write_scalar<std::uint64_t>(
                sink_, static_cast<std::uint64_t>(target_chunk_bytes_));
            chunk_count_field_ = tellp_u64(sink_, "zstd chunked chunk count field");
            detail::write_scalar<std::uint64_t>(sink_, 0u);
            table_offset_field_ = tellp_u64(sink_, "zstd chunked table offset field");
            detail::write_scalar<std::uint64_t>(sink_, 0u);

            pending_.reserve(target_chunk_bytes_);
        }

        ~ZstdChunkedOutputBuf() override = default;

        std::uint64_t encoded_bytes() const { return encoded_bytes_; }
        std::uint64_t decoded_bytes() const { return decoded_bytes_; }
        std::uint64_t target_chunk_bytes() const
        {
            return static_cast<std::uint64_t>(target_chunk_bytes_);
        }
        const std::vector<ZstdChunkEntryDisk>& chunks() const { return chunks_; }

        void finish()
        {
            if (finished_) return;
            flush_chunk();
            drain_compressed_chunks(true);

            const std::uint64_t table_offset = tellp_u64(sink_, "zstd chunk table");
            detail::write_scalar<std::uint32_t>(sink_, zstd_chunk_table_magic);
            detail::write_scalar<std::uint32_t>(sink_, 1u);
            detail::write_scalar<std::uint64_t>(
                sink_, static_cast<std::uint64_t>(chunks_.size()));
            for (const ZstdChunkEntryDisk& chunk : chunks_) {
                write_chunk_entry(sink_, chunk);
            }

            const std::uint64_t end_offset =
                tellp_u64(sink_, "zstd chunked section end");
            seekp_u64(sink_, decoded_size_field_, "zstd chunked decoded size field");
            detail::write_scalar<std::uint64_t>(sink_, decoded_bytes_);
            seekp_u64(sink_, chunk_count_field_, "zstd chunked chunk count field");
            detail::write_scalar<std::uint64_t>(
                sink_, static_cast<std::uint64_t>(chunks_.size()));
            seekp_u64(sink_, table_offset_field_, "zstd chunked table offset field");
            detail::write_scalar<std::uint64_t>(sink_, table_offset);
            seekp_u64(sink_, end_offset, "zstd chunked section end");

            encoded_bytes_ = end_offset - section_start_;
            finished_ = true;
        }

    protected:
        std::streamsize xsputn(const char* s, std::streamsize n) override
        {
            append_bytes(s, static_cast<std::size_t>(n));
            return n;
        }

        int overflow(int ch) override
        {
            if (ch == traits_type::eof()) return traits_type::not_eof(ch);
            const char c = static_cast<char>(ch);
            append_bytes(&c, 1);
            return ch;
        }

        int sync() override
        {
            sink_.flush();
            return sink_ ? 0 : -1;
        }

    private:
        static void write_chunk_entry(std::ostream& out,
                                      const ZstdChunkEntryDisk& entry)
        {
            detail::write_scalar<std::uint64_t>(out, entry.offset);
            detail::write_scalar<std::uint64_t>(out, entry.encoded_size);
            detail::write_scalar<std::uint64_t>(out, entry.decoded_offset);
            detail::write_scalar<std::uint64_t>(out, entry.decoded_size);
        }

        struct EncodedChunk {
            std::vector<char> bytes;
            std::uint64_t decoded_offset = 0;
            std::uint64_t decoded_size = 0;
        };

        struct PendingCompressedChunk {
            std::future<EncodedChunk> encoded;
        };

        static EncodedChunk compress_chunk(std::vector<char> decoded,
                                           std::uint64_t decoded_offset,
                                           int level)
        {
            const std::size_t bound = ZSTD_compressBound(decoded.size());
            std::vector<char> encoded(bound);
            const std::size_t compressed = ZSTD_compressCCtx(
                reusable_zstd_cctx(),
                encoded.data(),
                encoded.size(),
                decoded.data(),
                decoded.size(),
                level);
            if (ZSTD_isError(compressed)) {
                throw std::runtime_error(
                    std::string("failed while compressing zstd chunk: ") +
                    ZSTD_getErrorName(compressed));
            }
            encoded.resize(compressed);

            EncodedChunk chunk;
            chunk.bytes = std::move(encoded);
            chunk.decoded_offset = decoded_offset;
            chunk.decoded_size = static_cast<std::uint64_t>(decoded.size());
            return chunk;
        }

        void write_compressed_chunk(EncodedChunk chunk)
        {
            ZstdChunkEntryDisk entry;
            entry.offset = tellp_u64(sink_, "zstd chunk payload");
            entry.encoded_size = static_cast<std::uint64_t>(chunk.bytes.size());
            entry.decoded_offset = chunk.decoded_offset;
            entry.decoded_size = chunk.decoded_size;
            if (!chunk.bytes.empty()) {
                sink_.write(chunk.bytes.data(),
                            static_cast<std::streamsize>(chunk.bytes.size()));
                if (!sink_) throw std::runtime_error("failed while writing zstd chunk");
            }
            chunks_.push_back(entry);
        }

        void drain_compressed_chunks(bool force)
        {
            while (!pending_compressed_.empty() &&
                   (force || pending_compressed_.size() >= max_pending_compressed_)) {
                PendingCompressedChunk pending =
                    std::move(pending_compressed_.front());
                pending_compressed_.pop_front();
                write_compressed_chunk(pending.encoded.get());
            }
        }

        void append_bytes(const char* data, std::size_t size)
        {
            if (finished_) {
                throw std::runtime_error("attempted to write after finishing zstd chunked section");
            }
            while (size != 0) {
                const std::size_t space = target_chunk_bytes_ - pending_.size();
                const std::size_t n = std::min(space, size);
                pending_.insert(pending_.end(), data, data + n);
                data += n;
                size -= n;
                if (pending_.size() == target_chunk_bytes_) {
                    flush_chunk();
                }
            }
        }

        void flush_chunk()
        {
            if (pending_.empty()) return;

            const std::uint64_t decoded_offset = decoded_bytes_;
            decoded_bytes_ += static_cast<std::uint64_t>(pending_.size());
            std::vector<char> decoded = std::move(pending_);
            pending_.clear();
            pending_.reserve(target_chunk_bytes_);
            pending_compressed_.push_back(PendingCompressedChunk{
                spqri_chunk_pool().submit(
                    [decoded = std::move(decoded),
                     decoded_offset,
                     level = level_]() mutable {
                        return compress_chunk(
                            std::move(decoded), decoded_offset, level);
                    })
            });
            drain_compressed_chunks(false);
        }

        std::ostream& sink_;
        int level_ = 1;
        std::size_t target_chunk_bytes_ = 0;
        std::size_t max_pending_compressed_ = 1;
        std::vector<char> pending_;
        std::deque<PendingCompressedChunk> pending_compressed_;
        std::vector<ZstdChunkEntryDisk> chunks_;
        std::uint64_t section_start_ = 0;
        std::uint64_t decoded_size_field_ = 0;
        std::uint64_t chunk_count_field_ = 0;
        std::uint64_t table_offset_field_ = 0;
        std::uint64_t decoded_bytes_ = 0;
        std::uint64_t encoded_bytes_ = 0;
        bool finished_ = false;
    };

    class ZstdChunkedInputBuf : public std::streambuf {
    public:
        ZstdChunkedInputBuf(std::string path,
                            const SectionEntryDisk& section,
                            std::size_t prefetch_chunks)
            : path_(std::move(path)),
              section_(section),
              prefetch_chunks_(std::max<std::size_t>(1, prefetch_chunks))
        {
            const std::shared_ptr<const ZstdChunkTableData> table =
                load_cached_chunk_table(path_, section_);
            target_chunk_bytes_ = table->target_chunk_bytes;
            chunks_ = table->chunks;
            file_ = std::make_shared<PreadFile>(path_);
            setg(empty_, empty_, empty_);
            schedule_more();
        }

        std::uint64_t decoded_bytes() const { return decoded_loaded_; }

        static void cache_chunk_table(const std::string& path,
                                      const SectionEntryDisk& section,
                                      ZstdChunkTableData table)
        {
            const std::string key = chunk_table_cache_key(path, section);
            auto shared = std::make_shared<ZstdChunkTableData>(std::move(table));
            std::lock_guard<std::mutex> lock(chunk_table_cache_mutex());
            chunk_table_cache()[key] = std::move(shared);
        }

        static std::shared_ptr<const ZstdChunkTableData> load_cached_chunk_table(
            const std::string& path,
            const SectionEntryDisk& section)
        {
            const std::string key = chunk_table_cache_key(path, section);
            {
                std::lock_guard<std::mutex> lock(chunk_table_cache_mutex());
                auto it = chunk_table_cache().find(key);
                if (it != chunk_table_cache().end()) {
                    if (it->second) return it->second;
                }
            }

            std::shared_ptr<ZstdChunkTableData> table =
                std::make_shared<ZstdChunkTableData>(
                    load_chunk_table_uncached(path, section));
            {
                std::lock_guard<std::mutex> lock(chunk_table_cache_mutex());
                auto& slot = chunk_table_cache()[key];
                if (slot) return slot;
                slot = table;
            }
            return table;
        }

    protected:
        std::streamsize xsgetn(char* s, std::streamsize n) override
        {
            if (n <= 0) return 0;

            std::streamsize copied = 0;
            while (copied < n) {
                if (gptr() < egptr()) {
                    const std::streamsize buffered =
                        std::min<std::streamsize>(egptr() - gptr(), n - copied);
                    std::memcpy(s + copied, gptr(), static_cast<std::size_t>(buffered));
                    gbump(static_cast<int>(buffered));
                    copied += buffered;
                    continue;
                }

                const std::size_t remaining =
                    static_cast<std::size_t>(n - copied);
                const bool prefer_direct =
                    remaining >= std::max<std::size_t>(
                        1ull << 20,
                        checked_size(target_chunk_bytes_, "zstd chunk target size") / 2);
                if (prefer_direct) {
                    const std::size_t direct =
                        decode_direct_chunks(s + copied, remaining);
                    if (direct != 0) {
                        copied += static_cast<std::streamsize>(direct);
                        continue;
                    }
                }

                if (!load_next_chunk(!prefer_direct)) break;
            }
            return copied;
        }

        int underflow() override
        {
            if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
            if (!load_next_chunk()) return traits_type::eof();
            return traits_type::to_int_type(*gptr());
        }

    private:
        friend class IndexedSpqrTree;

        struct PendingChunk {
            std::uint64_t index = 0;
            std::future<std::vector<char>> bytes;
        };

        static std::uint64_t checked_add(std::uint64_t a,
                                         std::uint64_t b,
                                         const char* what)
        {
            if (a > std::numeric_limits<std::uint64_t>::max() - b) {
                throw std::runtime_error(std::string("overflow in ") + what);
            }
            return a + b;
        }

        static ZstdChunkEntryDisk read_chunk_entry(std::istream& in)
        {
            ZstdChunkEntryDisk entry;
            entry.offset = detail::read_scalar<std::uint64_t>(in, "zstd chunk offset");
            entry.encoded_size =
                detail::read_scalar<std::uint64_t>(in, "zstd chunk encoded size");
            entry.decoded_offset =
                detail::read_scalar<std::uint64_t>(in, "zstd chunk decoded offset");
            entry.decoded_size =
                detail::read_scalar<std::uint64_t>(in, "zstd chunk decoded size");
            return entry;
        }

        static std::string chunk_table_cache_key(const std::string& path,
                                                 const SectionEntryDisk& section)
        {
            std::ostringstream key;
            key << path << ':'
                << section.id << ':'
                << static_cast<unsigned>(section.codec) << ':'
                << section.offset << ':'
                << section.encoded_size << ':'
                << section.decoded_size;
#if defined(__unix__) || defined(__APPLE__)
            struct stat st;
            if (::stat(path.c_str(), &st) == 0) {
                key << ':'
                    << static_cast<unsigned long long>(st.st_dev) << ':'
                    << static_cast<unsigned long long>(st.st_ino) << ':'
                    << static_cast<unsigned long long>(st.st_size) << ':'
#if defined(__APPLE__)
                    << static_cast<long long>(st.st_mtimespec.tv_sec) << ':'
                    << static_cast<long long>(st.st_mtimespec.tv_nsec);
#else
                    << static_cast<long long>(st.st_mtim.tv_sec) << ':'
                    << static_cast<long long>(st.st_mtim.tv_nsec);
#endif
            }
#endif
            return key.str();
        }

        static std::mutex& chunk_table_cache_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        static std::unordered_map<
            std::string,
            std::shared_ptr<const ZstdChunkTableData>>& chunk_table_cache()
        {
            static std::unordered_map<std::string,
                                      std::shared_ptr<const ZstdChunkTableData>> cache;
            return cache;
        }

        static ZstdChunkTableData load_chunk_table_uncached(
            const std::string& path,
            const SectionEntryDisk& section)
        {
            const std::uint64_t section_end =
                checked_add(section.offset, section.encoded_size,
                            "zstd chunked section bounds");
            std::ifstream meta(path, std::ios::in | std::ios::binary);
            if (!meta) throw std::runtime_error("cannot open binary SPQR index: " + path);
            seekg_u64(meta, section.offset, "zstd chunked section");

            const std::uint32_t magic =
                detail::read_scalar<std::uint32_t>(meta, "zstd chunked section magic");
            if (magic != zstd_chunked_section_magic) {
                throw std::runtime_error("invalid zstd-chunked SPQR section in " + path);
            }
            const std::uint32_t version =
                detail::read_scalar<std::uint32_t>(meta, "zstd chunked section version");
            if (version != 1u) {
                throw std::runtime_error(
                    "unsupported zstd-chunked SPQR section version in " + path);
            }
            (void)detail::read_scalar<std::uint64_t>(meta, "zstd chunked section flags");
            const std::uint64_t decoded_size =
                detail::read_scalar<std::uint64_t>(meta, "zstd chunked decoded size");
            if (decoded_size != section.decoded_size) {
                throw std::runtime_error(
                    "zstd-chunked SPQR section decoded size mismatch in " + path);
            }
            ZstdChunkTableData table;
            table.target_chunk_bytes =
                detail::read_scalar<std::uint64_t>(meta, "zstd chunk target size");
            const std::uint64_t chunk_count =
                detail::read_scalar<std::uint64_t>(meta, "zstd chunk count");
            const std::uint64_t table_offset =
                detail::read_scalar<std::uint64_t>(meta, "zstd chunk table offset");

            if (table_offset < section.offset || table_offset >= section_end) {
                throw std::runtime_error(
                    "zstd-chunked SPQR section has invalid table offset in " + path);
            }
            seekg_u64(meta, table_offset, "zstd chunk table");
            const std::uint32_t table_magic =
                detail::read_scalar<std::uint32_t>(meta, "zstd chunk table magic");
            if (table_magic != zstd_chunk_table_magic) {
                throw std::runtime_error("invalid zstd chunk table in " + path);
            }
            const std::uint32_t table_version =
                detail::read_scalar<std::uint32_t>(meta, "zstd chunk table version");
            if (table_version != 1u) {
                throw std::runtime_error("unsupported zstd chunk table version in " + path);
            }
            const std::uint64_t table_count =
                detail::read_scalar<std::uint64_t>(meta, "zstd chunk table count");
            if (table_count != chunk_count) {
                throw std::runtime_error("zstd chunk table count mismatch in " + path);
            }
            const std::uint32_t count =
                graph_index::require_spqr_count(chunk_count, "zstd chunk count");
            table.chunks.reserve(count);
            std::uint64_t expected_decoded_offset = 0;
            for (std::uint32_t i = 0; i < count; ++i) {
                ZstdChunkEntryDisk entry = read_chunk_entry(meta);
                const std::uint64_t chunk_end =
                    checked_add(entry.offset, entry.encoded_size,
                                "zstd chunk payload bounds");
                if (entry.offset < section.offset || chunk_end > table_offset) {
                    throw std::runtime_error(
                        "zstd chunk payload is outside section bounds in " + path);
                }
                if (entry.decoded_offset != expected_decoded_offset) {
                    throw std::runtime_error(
                        "zstd chunk decoded offsets are not contiguous in " + path);
                }
                expected_decoded_offset =
                    checked_add(expected_decoded_offset, entry.decoded_size,
                                "zstd chunk decoded offsets");
                table.chunks.push_back(entry);
            }
            if (expected_decoded_offset != section.decoded_size) {
                throw std::runtime_error(
                    "zstd chunk decoded total mismatch in " + path);
            }

            const std::uint64_t table_end =
                tellg_u64(meta, "zstd chunk table end");
            if (table_end > section_end) {
                throw std::runtime_error(
                    "zstd chunk table extends beyond section bounds in " + path);
            }
            return table;
        }

        static std::vector<char> decompress_chunk_from_file(
            std::shared_ptr<const PreadFile> file,
            ZstdChunkEntryDisk entry)
        {
            std::vector<char> encoded(
                checked_size(entry.encoded_size, "zstd chunk encoded size"));
            if (!encoded.empty()) {
                file->read_exact(entry.offset,
                                 encoded.data(),
                                 encoded.size(),
                                 "zstd chunk encoded payload");
            }
            std::vector<char> decoded(
                checked_size(entry.decoded_size, "zstd chunk decoded size"));
            if (!decoded.empty()) {
                const std::size_t rc =
                    ZSTD_decompressDCtx(reusable_zstd_dctx(),
                                        decoded.data(),
                                        decoded.size(),
                                        encoded.data(),
                                        encoded.size());
                if (ZSTD_isError(rc)) {
                    throw std::runtime_error(
                        std::string("failed while decompressing zstd chunk: ") +
                        ZSTD_getErrorName(rc));
                }
                if (rc != decoded.size()) {
                    throw std::runtime_error("zstd chunk decoded size mismatch");
                }
            }
            return decoded;
        }

        static void decompress_chunk_into_buffer(
            std::shared_ptr<const PreadFile> file,
            ZstdChunkEntryDisk entry,
            char* out)
        {
            std::vector<char> encoded(
                checked_size(entry.encoded_size, "zstd chunk encoded size"));
            if (!encoded.empty()) {
                file->read_exact(entry.offset,
                                 encoded.data(),
                                 encoded.size(),
                                 "zstd chunk encoded payload");
            }
            const std::size_t decoded_size =
                checked_size(entry.decoded_size, "zstd chunk decoded size");
            if (decoded_size != 0) {
                const std::size_t rc =
                    ZSTD_decompressDCtx(reusable_zstd_dctx(),
                                        out,
                                        decoded_size,
                                        encoded.data(),
                                        encoded.size());
                if (ZSTD_isError(rc)) {
                    throw std::runtime_error(
                        std::string("failed while decompressing zstd chunk: ") +
                        ZSTD_getErrorName(rc));
                }
                if (rc != decoded_size) {
                    throw std::runtime_error("zstd chunk decoded size mismatch");
                }
            }
        }

        std::size_t decode_direct_chunks(char* out, std::size_t capacity)
        {
            if (!pending_.empty() || next_to_schedule_ != next_to_consume_) {
                return 0;
            }
            if (next_to_consume_ >= chunks_.size()) return 0;

            struct DirectChunk {
                std::future<void> done;
                std::uint64_t decoded_size = 0;
            };
            std::vector<DirectChunk> direct;
            direct.reserve(prefetch_chunks_);

            std::size_t bytes = 0;
            std::size_t count = 0;
            while (count < prefetch_chunks_ &&
                   next_to_consume_ + count < chunks_.size()) {
                const ZstdChunkEntryDisk entry =
                    chunks_[static_cast<std::size_t>(next_to_consume_) + count];
                const std::size_t decoded_size =
                    checked_size(entry.decoded_size, "zstd chunk decoded size");
                if (decoded_size == 0 || decoded_size > capacity - bytes) break;
                const std::shared_ptr<const PreadFile> file = file_;
                char* dst = out + bytes;
                direct.push_back(DirectChunk{
                    spqri_chunk_pool().submit(
                        [file, entry, dst]() {
                            decompress_chunk_into_buffer(file, entry, dst);
                        }),
                    entry.decoded_size
                });
                bytes += decoded_size;
                ++count;
            }
            if (direct.empty()) return 0;

            std::uint64_t decoded = 0;
            for (DirectChunk& chunk : direct) {
                chunk.done.get();
                decoded += chunk.decoded_size;
            }
            next_to_consume_ += direct.size();
            next_to_schedule_ = static_cast<std::size_t>(next_to_consume_);
            decoded_loaded_ += decoded;
            return bytes;
        }

        void schedule_more()
        {
            while (pending_.size() < prefetch_chunks_ &&
                   next_to_schedule_ < chunks_.size()) {
                const std::uint64_t index =
                    static_cast<std::uint64_t>(next_to_schedule_);
                const ZstdChunkEntryDisk entry = chunks_[next_to_schedule_++];
                const std::shared_ptr<const PreadFile> file = file_;
                pending_.push_back(PendingChunk{
                    index,
                    spqri_chunk_pool().submit(
                        [file, entry]() {
                            return decompress_chunk_from_file(file, entry);
                        })
                });
            }
        }

        bool load_next_chunk(bool refill_prefetch = true)
        {
            if (next_to_consume_ >= chunks_.size()) {
                if (decoded_loaded_ != section_.decoded_size) {
                    throw std::runtime_error(
                        "zstd-chunked SPQR section decoded size mismatch in " + path_);
                }
                return false;
            }
            if (pending_.empty()) schedule_more();
            if (pending_.empty()) return false;

            PendingChunk pending = std::move(pending_.front());
            pending_.pop_front();
            if (pending.index != next_to_consume_) {
                throw std::runtime_error("zstd chunk scheduler order mismatch");
            }
            current_ = pending.bytes.get();
            const ZstdChunkEntryDisk& entry = chunks_[next_to_consume_];
            if (current_.size() != checked_size(entry.decoded_size,
                                                "zstd chunk decoded size")) {
                throw std::runtime_error("zstd chunk decoded size mismatch");
            }
            ++next_to_consume_;
            decoded_loaded_ += entry.decoded_size;
            setg(current_.data(), current_.data(), current_.data() + current_.size());
            if (refill_prefetch) schedule_more();
            return !current_.empty();
        }

        std::string path_;
        SectionEntryDisk section_;
        std::size_t prefetch_chunks_ = 1;
        std::uint64_t target_chunk_bytes_ = 0;
        std::shared_ptr<PreadFile> file_;
        std::vector<ZstdChunkEntryDisk> chunks_;
        std::deque<PendingChunk> pending_;
        std::size_t next_to_schedule_ = 0;
        std::uint64_t next_to_consume_ = 0;
        std::uint64_t decoded_loaded_ = 0;
        std::vector<char> current_;
        char empty_[1] = {};
    };
#endif
