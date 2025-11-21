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
  BubbleFinder -g <graphPath> -o <outputPath> [options]

Options:
  --gfa                       Interpret the input graph as GFA (default: OFF)
  --superbubbles              Compute superbubbles
  --snarls                    Compute snarls
  -j <threadsNumber>          Number of threads
  -m <bytes>                  Stack size per thread in bytes
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

## Snarl algorithm correctness 

This repository also includes a brute-force implementation that computes all snarls in a naive way, which has been used to check the correctness of the main SPQR-tree-based implementation. This program gets built when building BubbleFinder from source (described above), and resides in the `build` directory after building from source.

To run this brute-force program computing snarls (on a given GFA file):
```
cd build 
./snarls_bf gfaGraphPath
```

We also include a generator of random graphs that automatically runs both implementations and compares their output. You can run this (for e.g. 100 random graphs) as:

```
python3 src/bruteforce.py --bruteforce-bin ./build/snarls_bf --bubblefinder-bin ./BubbleFinder --n-graphs 100
```
