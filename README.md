# BubbleFinder

## About
`BubbleFinder` is a program computing all snarls and superbubbles in genomic and pangenomic GFA graphs (i.e. bidirected graphs). `BubbleFinder` computes in linear-time a representation of all snarls whose size is linear in the size of the input graph. Superbubles are already known to be representable in linear time, as pairs of endpoints. 

`BubbleFinder` exploits the [SPQR trees](https://en.wikipedia.org/wiki/SPQR_tree) of the biconnected components of the undirected counterparts of the input bidirected graph, and traverses them efficiently to identify all snarls and superbubbles.

BubbleFinder supports two modes:

- `snarl`: computes **all** snarls and is supposed to replicate the behavior of [vg snarl](https://github.com/vgteam/vg) (when run with parameters -a -T). Note that `vg snarl` prunes some snarls to output only a linear-number of snarls; thus `BubbleFinder` finds more snarls than `vg snarls`.
- `superbubbles`: computes superbubbles in a (virtually) doubled representation of the bidirected graph and is supposed to replicate the behavior of [BubbleGun](https://github.com/fawaz-dabbaghieh/bubble_gun). Notice that BubbleGun also reports weak superbubbles, i.e. for a bubble with entry `s` and exit `t`, it also reports the structures which also have an edge from `t` to `s` (thus the interior of the bubble is not acyclic).

## Table of Contents
- [1. Installation](#1-installation)
- [2. Running](#2-running)
- [3. Example](#3-example)
- [4. Development](#4-development)
  - [GFA format and bidirected graphs](#gfa-format-and-bidirected-graphs)
  - [Snarl algorithm correctness](#snarl-algorithm-correctness)

# 1. Installation

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

# 2. Running 

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
  --gfa            GFA input (bidirected)
  --gfa-directed   GFA input interpreted as directed graph
  --graph          Internal .graph format (directed)
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
      Force .graph input (directed)
  --report-json <file>
      Write JSON metrics report
  -m <bytes>
      Stack size in bytes
  -h, --help
      Show this help message and exit
```

# 3. Example

Consider the bidirected graph below, which is encoded the file `example/tiny1.gfa`. 

![Tiny example - 1](example/tiny1.png)

You can run `BubbleFinder` on it as:
```
./BubbleFinder -g example/tiny1.gfa -o example/tiny1.snarls --gfa --snarls
```
After this, you should obtain the file `example/tiny1.snarls` with the following contents:
```
2
g+ k-
a+ d- f+ g-
```
The number of the first line is the number of lines in the file, and the following lines contains incidences such that *any* pair of incidences on each line is a snarl. So the snarls are {g+, k-} (from the second line in the file), and {a+, d-}, {a+, f+}, {a+, g-}, {d-, f+}, {d-, g-}, {f+, g-} (from the third line in the file).

# 4. Development

## GFA format and bidirected graphs

If you look at `example/tiny1.png` you'll notice that the bidirected edge `{a+, b+}` appearing in the graph image has been encoded as `L	a	+	b	-	0M`. This is because in GFA links are directed. So, the rule is that to compute snarls from a GFA file, for every link `a x b y` in the GFa file, (where x, y ∈ {+, -}), we flip the second sign `y` as `¬y`, and make an edge `{ax, b¬y}`. Then we compute snarls in this bidirected graph.

## Algorithm correctness 

This repository includes a brute-force implementation and a random test harness to validate both the snarl and superbubble algorithms.

### Brute-force implementation

A brute-force program is built together with BubbleFinder from source and resides in the `build` directory after compilation. It can compute snarls and, in a different mode, superbubbles on a given GFA file. In the examples below we refer to this binary as `./build/snarls_bf`.

To run the brute-force program on **snarls** for a given GFA file:

```bash
cd build
./snarls_bf gfaGraphPath
```

The exact output format is documented in `src/bruteforce.py`, which parses this output and compares it to BubbleFinder’s output. The same underlying brute-force implementation is also used to validate **superbubbles**: in practice, we usually drive it through the `src/bruteforce.py` script in `--superbubbles` mode (see below), which takes care of calling the binary and parsing its output in the appropriate way.

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
  --bruteforce-bin ./build/snarls_bf \
  --bubblefinder-bin ./BubbleFinder \
  --n-graphs 100
```

To stress-test the multi-threaded superbubble implementation and detect potential non-deterministic behaviour across different thread counts, you can run:

```bash
python3 src/bruteforce.py \
  --superbubbles \
  --bruteforce-bin ./build/snarls_bf \
  --bubblefinder-bin ./BubbleFinder \
  --n-graphs 1000 \
  --threads 1 4 8
```

Any divergence or error is reported, and if you pass `--keep-failing`, the script will save the corresponding GFA and BubbleFinder outputs in `/tmp` for manual inspection.