# BubbleFinder

## About
`BubbleFinder` is a program computing all snarls and superbubbles in genomic and pangenomic GFA graphs (i.e. bidirected graphs). `BubbleFinder` computes in linear time a representation of all snarls whose size is linear in the size of the input graph. Superbubbles are already known to be representable in linear time, as pairs of endpoints.

`BubbleFinder` exploits the [SPQR trees](https://en.wikipedia.org/wiki/SPQR_tree) of the biconnected components of the undirected counterparts of the input bidirected graph, and traverses them efficiently to identify all snarls and superbubbles.

BubbleFinder supports two main modes:

- `snarls`: computes **all** snarls and is supposed to replicate the behavior of [vg snarl](https://github.com/vgteam/vg) (when run with parameters `-a -T`). Note that `vg snarl` prunes some snarls to output only a linear number of snarls; thus `BubbleFinder` finds more snarls than `vg snarl`.
- `superbubbles`: computes superbubbles in a (virtually) doubled representation of the bidirected graph and is supposed to replicate the behavior of [BubbleGun](https://github.com/fawaz-dabbaghieh/bubble_gun). Since superbubbles are classically defined on **directed** graphs, BubbleFinder first runs the algorithm on this doubled directed representation, then projects the results back to unordered pairs of segment IDs (see [Orientation projection](#orientation-projection)). Notice that BubbleGun also reports weak superbubbles, i.e. for a bubble with entry `s` and exit `t`, it also reports the structures which also have an edge from `t` to `s` (thus the interior of the bubble is not acyclic).

## Table of Contents
- [1. Installation](#installation)
- [2. Running](#running)
  - [Output formats](#output-formats)
- [3. Example](#example)
- [4. Development](#development)
  - [GFA format and bidirected graphs](#gfa-format-and-bidirected-graphs)
  - [Orientation projection](#orientation-projection)
  - [Algorithm correctness](#algorithm-correctness)

# <a id="installation"></a>1. Installation

At the moment, building from source has been tested only on Linux:

```bash
git clone https://github.com/algbio/BubbleFinder && \
cd BubbleFinder && \
cmake -S . -B build && \
cmake --build build -j <NUM_THREADS> && \
mv build/BubbleFinder .
```

Replace `<NUM_THREADS>` with the number of parallel build jobs you want to use (for example `-j 8`). Omitting `-j` will build single‑threaded but more slowly.

Now `BubbleFinder` is in the root directory.

`conda` distributions for both Linux and macOS will be supported in the very near future.

# <a id="running"></a>2. Running 

To run BubbleFinder:

```text
Usage:
  ./BubbleFinder <command> -g <graphFile> -o <outputFile> [options]

Commands:
  superbubbles
      Bidirected superbubbles (GFA -> bidirected by default)
  directed-superbubbles
      Directed superbubbles (directed graph)
  snarls
      Snarls (typically on bidirected graphs from GFA)

Format options (input format):
  --gfa
      GFA input (bidirected).
  --gfa-directed
      GFA input interpreted as a directed graph.
  --graph
      .graph text format with one directed edge per line:
        • first line: two integers n and m
            - n = number of distinct node IDs declared
            - m = number of directed edges
        • next m lines: 'u v' (separated by whitespace),
            each describing a directed edge from u to v.
        • u and v are arbitrary node identifiers (strings
            without whitespace).
  If none of these is given, the format is auto-detected
  from the file extension (e.g. .gfa, .graph).

Compression:
  Compression is auto-detected from the file name suffix:
    .gz / .bgz  -> gzip
    .bz2        -> bzip2
    .xz         -> xz

General options:
  -g <file>
      Input graph file (possibly compressed)
  -o <file>
      Output file
  -j <threads>
      Number of threads
  --gfa
      Force GFA input (bidirected)
  --gfa-directed
      Force GFA input interpreted as directed graph
  --graph
      Force .graph text format (see 'Format options' above)
  --report-json <file>
      Write JSON metrics report
  -m <bytes>
      Stack size in bytes
  -h, --help
      Show this help message and exit
```

## <a id="output-formats"></a>2.1 Output formats

All commands produce plain text output with the same high‑level structure:

- **First line**: a single integer `N` – the number of result lines that follow.
- **Lines 2..N+1**: one result per line.

There are **two** output formats, depending on the command:

1. **Snarl format** (used by `snarls`)
2. **Pair format** (used by `superbubbles` and `directed-superbubbles`)

### 2.1.1 Snarl format (`snarls`)

Used by the `snarls` command.

- After the header, each line contains **at least two incidences**, separated by whitespace.
- An **incidence** is a segment (or node) ID followed by an orientation sign, e.g. `a+`, `d-`.

You can generate an example output on the tiny graph in `example/tiny1.gfa` with:

```bash
./BubbleFinder snarls -g example/tiny1.gfa -o example/tiny1.snarls --gfa
```

This produces a file `example/tiny1.snarls` containing:

```text
2
g+ k-
a+ d- f+ g-
```

Interpretation:

- The first line `2` means: **2 result lines follow**.
- On each subsequent line, **every unordered pair of incidences is a snarl**.
  - From `g+ k-` you get the single snarl `{g+, k-}`.
  - From `a+ d- f+ g-` you get all pairs:
    `{a+, d-}`, `{a+, f+}`, `{a+, g-}`, `{d-, f+}`, `{d-, g-}`, `{f+, g-}`.

So a single line encodes a whole *clique* of snarls.

### <a id="pair-format"></a>2.1.2 Pair format (`superbubbles`, `directed-superbubbles`)

Used by:

- `superbubbles` (bidirected graphs from GFA)
- `directed-superbubbles` (directed graphs: `--graph` or `--gfa-directed`)

Structure:

- After the header, each line contains **exactly two identifiers**:

```text
3
a b
e f
b e
```

Interpretation:

- For **`superbubbles`**:
  - Each line `u v` is an **unordered pair of segment IDs** `{u, v}`.
  - IDs are segment names from GFA `S` records (no `+/-` sign).
  - These pairs are obtained after orientation projection on the doubled directed graph (see [Orientation projection](#orientation-projection)).

- For **`directed-superbubbles`**:
  - Each line `s t` is an **ordered pair** describing a directed superbubble from entry `s` to exit `t` in the input directed graph.
  - Node identifiers are exactly those from the input:
    - for `.graph`, the node IDs used in the edge list,
    - for `--gfa-directed`, the oriented GFA segment identifiers (e.g. `a+`, `b-`).

# <a id="example"></a>3. Example

The repository contains a small toy graph in `example/tiny1.gfa`, visualized below:

![Tiny example - 1](example/tiny1.png)

This is exactly the graph used in the **snarl format example** in [Section 2.1.1](#output-formats).  
Running:

```bash
./BubbleFinder snarls -g example/tiny1.gfa -o example/tiny1.snarls --gfa
```

produces the snarl output explained in that section.

# <a id="development"></a>4. Development

## <a id="gfa-format-and-bidirected-graphs"></a>GFA format and bidirected graphs

If you look at `example/tiny1.png` you'll notice that the bidirected edge `{a+, b+}` appearing in the graph image has been encoded as:

```text
L	a	+	b	-	0M
```

This is because in GFA, links are directed. So, the rule is that to compute snarls from a GFA file, for every link

```text
a x b y
```

in the GFA file (where `x, y ∈ {+, -}`), we flip the second sign `y` as `¬y`, and make a bidirected edge `{a x, b ¬y}`. Then we compute snarls in this bidirected graph.

## <a id="orientation-projection"></a>Orientation projection

Superbubbles are classically defined on **directed** graphs, whereas GFA graphs are **bidirected**. In `superbubbles` mode, BubbleFinder therefore first converts the bidirected graph into a doubled directed graph, where each segment `v` has two oriented copies `v+` and `v-`, and superbubbles are detected between oriented endpoints (e.g. `(a+, e-)` and its mirror `(e+, a-)`).

To report results at the level of segments, independently of the arbitrary orientation chosen in the doubled graph, we apply an **orientation projection**:

- mirror pairs like `(a+, e-)` and `(e+, a-)` are treated as the same bubble;
- we then drop the `+/-` signs and sort the two segment names.

The final output is a single unordered pair of segment IDs, e.g. `a e`. This process is illustrated below.

<p align="center">
  <img src="example/projection-example.svg" alt="Orientation projection example">
</p>

In this example, the directed graph on segments has three superbubbles with endpoints `(a, b)`, `(b, e)` and `(e, f)`. After running the superbubble algorithm on the doubled graph and applying the orientation projection, BubbleFinder reports exactly these three pairs in its **pair format output** (see [Pair format](#pair-format)).

## <a id="algorithm-correctness"></a>Algorithm correctness 

This repository includes a brute-force implementation and a random test harness to validate both the snarl and superbubble algorithms.

### Brute-force implementation

A brute-force program is built together with BubbleFinder from source and resides in the `build` directory after compilation. It can compute snarls and, in a different mode, superbubbles on a given GFA file. In the examples below we refer to this binary as `./build/snarls_bf`.

To run the brute-force program on **snarls** for a given GFA file:

```bash
cd build
./snarls_bf gfaGraphPath
```

The brute-force program outputs results in a format consumed by `src/bruteforce.py`, which then compares them with BubbleFinder’s output. The same brute-force engine is also used for superbubble validation via the `--superbubbles` mode, which takes care of running the binary and parsing its output.

### Random testing with `src/bruteforce.py`

We also include a generator of random graphs and a driver script that compares the brute-force implementation with BubbleFinder. The script is `src/bruteforce.py`, and it can operate in two modes:

- **Snarls** (default)
- **Superbubbles** (via `--superbubbles`)

#### Snarls

To run the random tests on snarls (for example, on 100 random graphs):

```bash
python3 src/bruteforce.py \
  --bruteforce-bin ./build/snarls_bf \
  --bubblefinder-bin ./BubbleFinder \
  --n-graphs 100
```

This will:

- generate random GFA graphs,
- run the brute-force snarl finder,
- run BubbleFinder in snarl mode,
- compare their outputs (missing / extra pairs / segfaults),
- and additionally check that BubbleFinder’s output is consistent across different thread counts if you pass several values to `--threads`.

#### Superbubbles

To run the same style of tests for superbubbles, pass the `--superbubbles` flag:

```bash
python3 src/bruteforce.py \
  --superbubbles \
  --bruteforce-bin ./build/superbubbles_bf \
  --bubblefinder-bin ./BubbleFinder \
  --n-graphs 100
```

To run superbubble tests with multiple thread configurations and check result consistency, you can run:

```bash
python3 src/bruteforce.py \
  --superbubbles \
  --bruteforce-bin ./build/superbubbles_bf \
  --bubblefinder-bin ./BubbleFinder \
  --n-graphs 100 \
  --threads 1 4 8
```

Any divergence or error is reported, and if you pass `--keep-failing`, the script will save the corresponding GFA and BubbleFinder outputs in `/tmp` for manual inspection.
