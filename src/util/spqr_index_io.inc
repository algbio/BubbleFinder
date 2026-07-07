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
