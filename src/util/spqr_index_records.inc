    // Read and write the flat record arrays stored in .spqri sections.

    static void write_raw_bytes(std::ostream& out, const void* data, std::size_t bytes)
    {
        const char* p = static_cast<const char*>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(1ull << 30));
            out.write(p, static_cast<std::streamsize>(chunk));
            if (!out) throw std::runtime_error("failed while writing binary SPQR index data");
            p += chunk;
            bytes -= chunk;
        }
    }

    static void read_raw_bytes(std::istream& in, void* data, std::size_t bytes, const char* what)
    {
        char* p = static_cast<char*>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(1ull << 30));
            in.read(p, static_cast<std::streamsize>(chunk));
            if (!in) throw std::runtime_error(std::string("truncated binary SPQR index data: ") + what);
            p += chunk;
            bytes -= chunk;
        }
    }
