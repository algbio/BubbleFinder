# BubbleFinder

[![CI](https://github.com/algbio/BubbleFinder/actions/workflows/bruteforce.yml/badge.svg)](https://github.com/algbio/BubbleFinder/actions/workflows/bruteforce.yml)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/bubblefinder)](https://anaconda.org/bioconda/bubblefinder)
[![GitHub release](https://img.shields.io/github/v/release/algbio/BubbleFinder.svg?style=flat-square)](https://github.com/algbio/BubbleFinder/releases/latest)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)

`BubbleFinder` computes **all snarls**, **superbubbles**, and **ultrabubbles** in genomic and pangenomic **GFA** and **GBZ** graphs (i.e. bidirected graphs).

All algorithms run in **linear time** in the size of the input graph (`O(|V|+|E|)`). Ultrabubbles are computed using two modes: **oriented mode** (default), which orients the bidirected graph and reduces to directed weak superbubbles (requires at least one tip or one cut vertex per connected component), and **doubled mode** (`--doubled`), which builds a doubled directed graph with no restriction on connected components, but uses more RAM.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Commands overview](#commands-overview)
- [Running BubbleFinder](#running-bubblefinder)
  - [Input data](#input-data)
  - [Command-line options](#command-line-options)
- [Output format](#output-format)
- [References](#references)

**Additional resources (Wiki):**
- [Flowchart](../../wiki/Flowchart), which details execution paths for all commands
- [Internals](../../wiki/Internals), covering GFA/bidirected graph representation, orientation projection, and theoretical background
- [Validation](../../wiki/Validation), describing bruteforce testing and the random test harness

---

### Snarls and superbubbles via SPQR trees

BubbleFinder first builds the undirected version of the input bidirected graph, then uses the [SPQR trees](https://en.wikipedia.org/wiki/SPQR_tree) of its biconnected components to identify all snarls and superbubbles.

> [!IMPORTANT]
> `snarls` computes **all** snarls and aims to replicate the behavior of [`vg snarls -a -T`](https://github.com/vgteam/vg), **but** `vg` outputs only a pruned, linear-size *snarl decomposition*.  
> Therefore, `BubbleFinder` may output **more** snarls than `vg snarls`.

> [!NOTE]
> **Empirical performance (snarls & superbubbles).** Benchmarks and theory are reported in [Sena, Politov et al., 2025](#ref-sena-politov2025).
> - **Snarls:** BubbleFinder is consistently faster than [`vg snarls -a -T`](https://github.com/vgteam/vg) on the PGGB graphs (up to ~2× faster on larger graphs and ~3× on the smallest one). On human chromosome graphs (Chromosome 1/10/22), BubbleFinder can be up to ~2× slower end-to-end in a single-threaded run due to preprocessing (BC/SPQR tree building), but benefits from multi-threading (up to ~4× speedup at 16 threads in those datasets).
> - **Superbubbles:** BubbleFinder runs in similar times as [BubbleGun](https://github.com/fawaz-dabbaghieh/bubble_gun) on small graphs, and is about ~10× faster on larger graphs; in particular, BubbleGun hit a **>3h timeout** on Chromosome 1/10/22, while BubbleFinder completed in minutes in our benchmarks.

### Ultrabubbles via linear-time orientation + reduction to weak superbubbles

Ultrabubbles use a different approach (not SPQR-based). BubbleFinder first **orients** the bidirected graph into a directed graph using a DFS-based procedure, then runs a linear-time directed weak superbubble algorithm on the result and maps the output back to ultrabubbles in the original bidirected graph.

> [!NOTE]
> **Empirical performance (ultrabubbles).** In our ultrabubble benchmarks, BubbleFinder consistently outperformed `vg` across all tested datasets. On the HPRC graphs in GBZ format, excluding parsing time, BubbleFinder achieves speedups of **19–26×** over `vg`. On HPRC v2.0 CHM13 (232 individuals), after parsing, BubbleFinder completes in **under 3 minutes** while `vg` requires **more than one hour**, using **four times less RAM** (24.8 GiB vs 101.8 GiB). On GFA input, BubbleFinder is **~200× faster** than [BubbleGun](https://github.com/fawaz-dabbaghieh/bubble_gun) on the HPRC v1.1 graph (47 individuals).  
> A dedicated preprint describing this method, its correctness, and benchmarks is available: [Harviainen, Sena, Moumard et al., 2026](#ref-harviainen2026).

---

## Quickstart

### Prebuilt Linux binary

Download the latest release:

https://github.com/algbio/BubbleFinder/releases/latest

```bash
./BubbleFinder --help
./BubbleFinder snarls -g example/tiny1.gfa -o tiny1.snarls
```

### Conda / Bioconda

```bash
conda create -n bubblefinder_env -c conda-forge -c bioconda bubblefinder
conda activate bubblefinder_env
./BubbleFinder --help
```

### Build from source (Linux)

```bash
git clone --recurse-submodules https://github.com/algbio/BubbleFinder && \
cd BubbleFinder && \
cmake -S . -B build && \
cmake --build build -j <NUM_THREADS> && \
mv build/BubbleFinder .
```

Replace `<NUM_THREADS>` with the number of parallel build jobs (e.g. `-j 8`). Omitting `-j` builds single-threaded.

**Dependencies** are handled automatically by the build system:
- **[OGDF](https://github.com/ogdf/ogdf)** is fetched and built via CMake FetchContent.
- **[zstd](https://github.com/facebook/zstd)** is detected on the system. If not found, it is automatically fetched and built from source.
- **[OpenSSL](https://www.openssl.org/)** (`libcrypto`) must be available on the system (pre-installed on most Linux distributions).
- GBZ support pulls in four submodules, all under `external/gbz/` and built automatically:
  - **[gbwtgraph](https://github.com/jltsiren/gbwtgraph)** is the GBZ/GBWTGraph library
  - **[gbwt](https://github.com/jltsiren/gbwt)** is the GBWT index (required by gbwtgraph)
  - **[sdsl-lite](https://github.com/vgteam/sdsl-lite)** (vgteam fork), which provides low-level data structures (required by gbwt)
  - **[libhandlegraph](https://github.com/vgteam/libhandlegraph)** is the handle graph interface (required by gbwtgraph)
---

## Commands overview

| Command | Typical input | Output endpoints | Notes |
|---|---|---|---|
| `snarls` | bidirected GFA / GBZ | oriented incidences (`a+`, `d-`) | may output cliques |
| `superbubbles` | bidirected GFA / GBZ (default) or directed (`--directed`) | segment IDs (`a`, `e`) in bidirected mode; oriented IDs (`a+`, `e-`) in directed mode | computed on doubled directed graph + orientation projection (bidirected) or directly (directed) |
| `ultrabubbles` | bidirected GFA / GBZ | oriented incidences | oriented mode (default): ≥ 1 tip or cut vertex per CC; doubled mode (`--doubled`): no restriction |
| `spqr-tree` | **GFA / GBZ only** | `.spqr` v0.4 | connected components + BC-tree + SPQR decomposition |

All commands except `spqr-tree` exclude trivial bubbles by default (use `-T` to include them), and are validated against a brute-force implementation on randomly generated graphs (see [Validation](../../wiki/Validation) on the Wiki).

For a detailed walkthrough of all execution paths, see the [Flowchart](../../wiki/Flowchart) on the Wiki.

---

## Running BubbleFinder

```text
./BubbleFinder <command> -g <graphFile> -o <outputFile> [options]
```

Available commands:
- `superbubbles` find superbubbles (bidirected by default, use `--directed` for directed mode)
- `snarls` find snarls (typically on bidirected graphs from GFA)
- `ultrabubbles` find ultrabubbles (oriented mode by default, use `--doubled` for doubled graph mode)
- `spqr-tree` output the connected components, BC-tree and SPQR decomposition in `.spqr` v0.4 format

> [!WARNING]
> In oriented mode (default), `ultrabubbles` requires **at least one tip or one cut vertex per connected component** in the input graph (otherwise it will fail). Use `--doubled` if your graph has tipless and cut-vertex-free connected components.

### Input data

| Extension | Format | Description |
|---|---|---|
| `.gfa` / `.gfa1` | GFA1 | Graphical Fragment Assembly format |
| `.gbz` | GBZ | vg/gbwtgraph binary format |
| `.graph` | BubbleFinder text | Simple directed edge list (see below) |

BubbleFinder `.graph` text format:
- first line: two integers `n` (number of node IDs) and `m` (number of directed edges)
- next `m` lines: `u v` (one directed edge per line)
- `u` and `v` are arbitrary node identifiers (strings without whitespace)

Force the input format with `--gfa`, `--gfa-directed`, or `--graph`. Input files can be compressed (gzip, bzip2, xz), auto-detected from the file suffix.

> [!NOTE]
> `spqr-tree` currently requires **GFA or GBZ input**.

### Command-line options

| Option | Description |
|---|---|
| `-g <file>` | Input graph file (possibly compressed) |
| `-o <file>` | Output file |
| `-j <threads>` | Number of threads |
| `--gfa` | Force GFA input (bidirected) |
| `--gfa-directed` | Force GFA input interpreted as directed graph |
| `--graph` | Force `.graph` text format |
| `--directed` | Interpret graph as directed (for `superbubbles`) |
| `--doubled` | Use doubled-graph algorithm (for `ultrabubbles`) |
| `-T`, `--include-trivial` | Include trivial bubbles in output |
| `--clsd-trees <file>` | Write ultrabubble hierarchy to `<file>` (`ultrabubbles` only) |
| `--report-json <file>` | Write JSON metrics report |
| `-m <bytes>` | Stack size in bytes |
| `-h`, `--help` | Show help and exit |

---

## Output format

All commands write plain text to the file given by `-o <outputFile>`. The first line is a single integer `N` (the number of result lines that follow), and lines 2 through N+1 each contain one result.

Each result line encodes one or more **unordered pairs of endpoints**. What an "endpoint" looks like depends on the command: `snarls` and `ultrabubbles` use **oriented incidences** (e.g. `a+`, `d-`), `superbubbles` in bidirected mode uses **segment IDs without orientation** (e.g. `a`, `e`), and `superbubbles --directed` uses **oriented IDs** (e.g. `a+`, `e-`).

<details>
<summary><strong>Snarls</strong></summary>

By default, **trivial snarls** are excluded. Use `-T` / `--include-trivial` to include them.

**With `-T`**: each line contains at least two incidences. A line with `k ≥ 2` incidences encodes **all unordered pairs** among them (clique representation).

Example on `example/tiny1.gfa`:

```bash
./BubbleFinder snarls -T -g example/tiny1.gfa -o example/tiny1.snarls --gfa
```

```text
2
g+ k-
a+ d- f+ g-
```

- `g+ k-` → single pair `{g+, k-}`.
- `a+ d- f+ g-` → all pairs: `{a+, d-}`, `{a+, f+}`, `{a+, g-}`, `{d-, f+}`, `{d-, g-}`, `{f+, g-}`.

**Without `-T`** (default): cliques are expanded, trivial pairs filtered, each line contains exactly two oriented incidences.

</details>

<details>
<summary><strong>Superbubbles</strong></summary>

In **bidirected mode** (default), each result line contains exactly two segment IDs (no orientation):

```text
3
a b
e f
b e
```

These pairs are obtained after running the superbubble algorithm on the doubled directed graph and applying the orientation projection (see [Internals](../../wiki/Internals) on the Wiki).

In **directed mode** (`--directed`), each result line contains two oriented IDs:

```text
3
a+ b-
e+ f-
b+ e-
```

</details>

<details>
<summary><strong>Ultrabubbles</strong></summary>

A flat list of endpoint pairs where each endpoint is an oriented incidence (`segmentID+` / `segmentID-`):

```text
N
a+ d-
g+ k-
...
```

Both oriented mode (default) and doubled mode (`--doubled`) produce the same output format.

To also output the **hierarchical nesting structure**, use `--clsd-trees <file>`. Each line in that file is a rooted tree in parenthesized form:

- leaf bubble: `<X,Y>`
- internal bubble: `(child1,...,childk)<X,Y>`

where `X` and `Y` are oriented incidences such as `a+` or `d-`.

</details>

<details>
<summary><strong>SPQR-tree</strong></summary>

The `spqr-tree` command writes a `.spqr` file according to the [SPQR tree file format](https://github.com/sebschmi/SPQR-tree-file-format) specification (version **v0.4**). BubbleFinder writes the header:

```text
H v0.4 https://github.com/sebschmi/SPQR-tree-file-format
```

For details on line types and semantics, refer to the specification repository.

</details>

---

## References

- <a id="ref-sena-politov2025"></a> Francisco Sena, Aleksandr Politov, Corentin Moumard, Manuel Cáceres, Sebastian Schmidt, Juha Harviainen, Alexandru I. Tomescu. *Identifying all snarls and superbubbles in linear-time, via a unified SPQR-tree framework*. arXiv:2511.21919 (2025). https://arxiv.org/abs/2511.21919

- <a id="ref-gartner2019direct"></a> Fabian Gärtner, Peter F. Stadler. *Direct superbubble detection*. Algorithms 12(4):81, 2019. DOI: 10.3390/a12040081. https://www.mdpi.com/1999-4893/12/4/81

- <a id="ref-gbz"></a> Jouni Sirén, Benedict Paten. *GBZ file format for pangenome graphs*. Bioinformatics 38(22):5012–5018, 2022. DOI: 10.1093/bioinformatics/btac656. https://academic.oup.com/bioinformatics/article/38/22/5012/6731924

- <a id="ref-vg"></a> vg toolkit (GitHub): https://github.com/vgteam/vg

- <a id="ref-bubblegun"></a> BubbleGun (GitHub): https://github.com/fawaz-dabbaghieh/bubble_gun

- <a id="ref-harviainen2026"></a> Juha Harviainen, Francisco Sena, Corentin Moumard, Aleksandr Politov, Sebastian Schmidt, Alexandru I. Tomescu. *Scalable computation of ultrabubbles in pangenomes by orienting bidirected graphs*. bioRxiv 2026.03.28.714704 (2026). https://www.biorxiv.org/content/10.64898/2026.03.28.714704
