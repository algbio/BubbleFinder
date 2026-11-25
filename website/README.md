# BubbleFinder website

A static website with no build step or external JavaScript dependencies.

From the repository root, preview it with:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Then open http://127.0.0.1:8000/ in a browser.

- `index.html`: overview, bubble figure and publications.
- `get-started.html`: installation and a downloadable GFA example.
- `documentation.html`: method selection, flags, output and references.
- `spqr-rust.html`: library overview, decomposition figure and references.
- `spqr-rust-documentation.html`: manual contents and topic index.
- `spqr-rust-guide.html`: ordinary Rust graph/tree usage.
- `spqr-rust-compression.html`: compression, contracts, macros and reconstruction.
- `spqr-rust-interfaces.html`: wide IDs, C/C++, FFI, CLI and runtime settings.
- `spqr-rust-api.html`: reference index, linked to generated API pages and full headers under spqr-rust-api/.
- `assets/styles.css` and `assets/site.js`: shared presentation and interactions.


The bubble illustration is copied from `../docs/bubble-examples.svg`, generated from the accompanying TikZ source. Copy the SVG again after changing that figure.

The sites are published together with GitHub Pages:

- BubbleFinder: https://algbio.github.io/BubbleFinder/
- SPQR-Rust: https://algbio.github.io/BubbleFinder/spqr-rust.html

Relative links support this project path. `.github/workflows/pages.yml` publishes `website/` when its files change on `main`. The workflow can also be run manually.

The layout takes its reference from the historical BWA website (https://bio-bwa.sourceforge.net/). Its stylesheet (https://bio-bwa.sourceforge.net/style.css) was served with Last-Modified: 2010-07-02 when checked on 2026-11-18. The BubbleFinder layout is implemented independently, with a wider figure area and responsive behavior.

The SPQR example figure is generated from assets/spqr-example.tex. The example in examples/spqr-example.rs was compiled and run using Rust 1.93.0 and SPQR-Rust commit f715f217de367df04e8dd339b63a9bae653a7eba. It produced S: 1, P: 1, R: 2 and passed the library structural verifier. Build artifacts were kept outside this checkout.

The documentation examples were checked against the same SPQR-Rust commit on 2026-11-19. Rust component access, tree traversal, raw/reduced verification and .spqr round trips (including a self-loop) passed. The C++ example compiled, linked against the shared library and ran successfully. All checks used temporary files outside this checkout.

The full API reference was generated from the same source commit using Rust 1.93.0. Compression/reconstruction, protected masks, self-loops, fully reducible graphs, sparse 64-bit compression, per-block decomposition and C FFI examples were checked. The command-line example was also run. The reference can be regenerated with tools/build_spqr_reference.py, using a fresh temporary output directory and a local source checkout.
