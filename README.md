# BubbleFinder

## About
`BubbleFinder` is a program computing all snarls and superbubbles in genomic and pangenomic GFA graphs (i.e. bidirected graphs). `BubbleFinder` computes in linear-time a representation of all snarls whose size is linear in the size of the input graph. Superbubles are already known to be representable in linear time, as pairs of endpoints. 

`BubbleFinder` exploits the [SPQR trees](https://en.wikipedia.org/wiki/SPQR_tree) of the biconnected components of the undirected counterparts of the input bidirected graph, and traverses them efficiently to identify all snarls and superbubbles.

BubbleFinder supports two modes:

- `snarl`: computes **all** snarls and is supposed to replicate the behavior of [vg snarl](https://github.com/vgteam/vg) (when run with parameters -a -T). Note that `vg snarl` prunes some snarls to output only a linear-number of snarls; thus `BubbleFinder` finds more snarls than `vg snarls`.
- `superbubbles`: computes superbubbles in a (virtually) doubled representation of the bidirected graph and is supposed to replicate the behavior of [BubbleGun](https://github.com/fawaz-dabbaghieh/bubble_gun). Notice that BubbleGun also reports weak superbubbles, i.e. for a bubble with entry `s` and exit `t`, it also reports the structures which also have an edge from `t` to `s` (thus the interior of the bubble is not acyclic).

## Table of Contents
- [1. Installation](#installation)
- [2. Running](#running)
- [3. Example](#example)
- [4. Development](#development)
  - [GFA format and bidirected graphs](#gfa-format-and-bidirected-graphs)
  - [Algorithm correctness](#algorithm-correctness)

# <a id="installation"></a>1. Installation

At the moment, building from source has been tested only on linux:

```
git clone https://github.com/algbio/BubbleFinder && \
cd BubbleFinder && \
cmake -S . -B build && \
cmake --build build && \
mv build/BubbleFinder .
```

Now `BubbleFinder` is in the root directory.

`conda` distributions for both linux and macos will be supported in the very near future.

# <a id="running"></a>2. Running 

To run BubbleFinder:
```
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

# <a id="example"></a>3. Example

Consider the bidirected graph below, which is encoded the file `example/tiny1.gfa`. 

![Tiny example - 1](example/tiny1.png)

You can run `BubbleFinder` on it as:
```
./BubbleFinder snarls -g example/tiny1.gfa -o example/tiny1.snarls --gfa
```
After this, you should obtain the file `example/tiny1.snarls` with the following contents:
```
2
g+ k-
a+ d- f+ g-
```
The number of the first line is the number of lines in the file, and the following lines contains incidences such that *any* pair of incidences on each line is a snarl. So the snarls are {g+, k-} (from the second line in the file), and {a+, d-}, {a+, f+}, {a+, g-}, {d-, f+}, {d-, g-}, {f+, g-} (from the third line in the file).

# <a id="development"></a>4. Development

## <a id="gfa-format-and-bidirected-graphs"></a>GFA format and bidirected graphs

If you look at `example/tiny1.png` you'll notice that the bidirected edge `{a+, b+}` appearing in the graph image has been encoded as `L	a	+	b	-	0M`. This is because in GFA links are directed. So, the rule is that to compute snarls from a GFA file, for every link `a x b y` in the GFa file, (where x, y ∈ {+, -}), we flip the second sign `y` as `¬y`, and make an edge `{ax, b¬y}`. Then we compute snarls in this bidirected graph.

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
