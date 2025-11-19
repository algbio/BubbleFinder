#!/usr/bin/env python3
"""
This script generates random GFA graphs, runs two different snarl finding
methods (snarls_bf and BubbleFinder), and compares their outputs.
Its purpose is to track potential mismatch, errors, or non deterministic
behaviour under several threads counts
"""

import argparse
import random
import subprocess
import tempfile
from pathlib import Path


def generate_random_gfa(path, n_nodes, n_edges, seed=None):
    rng = random.Random(seed)

    nodes = list(range(1, n_nodes + 1))
    edges = set()

    with open(path, "w") as f:
        for n in nodes:
            f.write(f"S {n} x\n")

        while len(edges) < n_edges:
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u == v:
                continue
            du = rng.choice(["+", "-"])
            dv = rng.choice(["+", "-"])
            edge = (u, du, v, dv)
            if edge in edges:
                continue
            edges.add(edge)

        for (u, du, v, dv) in edges:
            f.write(f"L {u} {du} {v} {dv} x\n")


def parse_snarls_bruteforce_output(stdout):
    snarls = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        ep1, ep2 = parts
        n1, s1 = int(ep1[:-1]), ep1[-1]
        n2, s2 = int(ep2[:-1]), ep2[-1]
        e1 = (n1, s1)
        e2 = (n2, s2)
        snarl = tuple(sorted([e1, e2]))
        snarls.add(snarl)
    return snarls


def parse_snarls_bubblefinder_file(path: Path):
    snarls = set()

    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return snarls

    start_idx = 0
    first_tokens = lines[0].split()
    if len(first_tokens) == 1 and first_tokens[0].isdigit():
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.split()
        endpoints = []

        for ep in parts:
            ep = ep.strip()
            if not ep:
                continue
            if len(ep) < 2:
                continue
            try:
                node = int(ep[:-1])
                sign = ep[-1]
            except ValueError:
                continue
            if sign not in {"+", "-"}:
                continue
            endpoints.append((node, sign))

        k = len(endpoints)
        if k < 2:
            continue
        for i in range(k):
            for j in range(i + 1, k):
                e1 = endpoints[i]
                e2 = endpoints[j]
                snarls.add(tuple(sorted((e1, e2))))

    return snarls


def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def run_snarls_bf(bruteforce_bin, gfa_path):
    res = subprocess.run(
        [bruteforce_bin, str(gfa_path)],
        capture_output=True,
        check=True,
    )
    stdout = _decode_bytes(res.stdout)
    return parse_snarls_bruteforce_output(stdout)


def run_bubblefinder(bf_bin, gfa_path, out_path, threads):
    cmd = [
        bf_bin,
        "-g",
        str(gfa_path),
        "--gfa",
        "-o",
        str(out_path),
        "--snarls",
        "-j",
        str(threads),
    ]
    res = subprocess.run(cmd, capture_output=True)

    stdout = _decode_bytes(res.stdout)
    stderr = _decode_bytes(res.stderr)

    if res.returncode != 0:
        raise subprocess.CalledProcessError(
            res.returncode,
            res.args,
            output=stdout,
            stderr=stderr,
        )

    return parse_snarls_bubblefinder_file(out_path)


def compare_snarls(s1, s2):
    missing = s1 - s2
    extra = s2 - s1
    return missing, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bruteforce-bin", default="./snarls_bf",
                        help="Path to brute-force binary")
    parser.add_argument("--bubblefinder-bin", default="./BubbleFinder",
                        help="Path to BubbleFinder binary")
    parser.add_argument("--n-graphs", type=int, default=10000,
                        help="Number of random graphs to test")
    parser.add_argument("--min-nodes", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--min-edges", type=int, default=2)
    parser.add_argument("--max-edges", type=int, default=20)
    parser.add_argument("--seed", type=int, default=440)
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1],
        help="List of thread counts to test for BubbleFinder (e.g. --threads 1 7)"
    )
    parser.add_argument("--keep-failing", action="store_true",
                        help="Save failing .gfa files (and BF outputs) in /tmp")
    parser.add_argument(
        "--max-report-graphs",
        type=int,
        default=5,
        help="Max number of failing graphs whose full GFA is printed at the end"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    threads_list = sorted(set(args.threads))

    n_fail = 0
    failures = []

    for i in range(args.n_graphs):
        n_nodes = rng.randint(args.min_nodes, args.max_nodes)
        n_edges = rng.randint(args.min_edges, args.max_edges)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gfa_path = tmpdir / "input.gfa"
            generate_random_gfa(gfa_path, n_nodes, n_edges, seed=rng.randint(0, 10**9))

            graph_failed = False
            reasons = set()

            try:
                snarls_bf = run_snarls_bf(args.bruteforce_bin, gfa_path)
            except subprocess.CalledProcessError as e:
                graph_failed = True
                reasons.add("bruteforce_error")
                print(f"[Graph {i}] ERROR in brute-force (returncode={e.returncode})")
                if e.output:
                    print("  [bruteforce stdout]:")
                    print("  <<<")
                    print(e.output.rstrip())
                    print("  >>>")
                if e.stderr:
                    print("  [bruteforce stderr]:")
                    print("  <<<")
                    print(e.stderr.rstrip())
                    print("  >>>")

                sample_gfa = None
                if len(failures) < args.max_report_graphs:
                    sample_gfa = gfa_path.read_text()
                if args.keep_failing:
                    dest = Path("/tmp") / f"fail_graph_{i}.gfa"
                    dest.write_text(gfa_path.read_text())
                    print(f"  GFA saved to {dest}")

                    dest_stdout = Path("/tmp") / f"fail_graph_{i}_bruteforce.stdout.txt"
                    dest_stderr = Path("/tmp") / f"fail_graph_{i}_bruteforce.stderr.txt"
                    dest_stdout.write_text(e.output or "")
                    dest_stderr.write_text(e.stderr or "")
                    print(f"  bruteforce stdout saved to {dest_stdout}")
                    print(f"  bruteforce stderr saved to {dest_stderr}")

                failures.append({
                    "index": i,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "reasons": sorted(reasons),
                    "threads": threads_list,
                    "gfa": sample_gfa,
                })
                n_fail += 1
                continue

            snarls_by_threads = {}
            bf_error = False
            for t in threads_list:
                out_path = tmpdir / f"out_t{t}.sbfind"
                try:
                    snarls_by_threads[t] = run_bubblefinder(
                        args.bubblefinder_bin, gfa_path, out_path, threads=t
                    )
                except subprocess.CalledProcessError as e:
                    graph_failed = True
                    bf_error = True
                    reasons.add("bubblefinder_error")
                    print(f"[Graph {i}] ERROR in BubbleFinder (threads={t}, returncode={e.returncode})")

                    if e.output:
                        print("  [BubbleFinder stdout]:")
                        print("  <<<")
                        print(e.output.rstrip())
                        print("  >>>")
                    if e.stderr:
                        print("  [BubbleFinder stderr]:")
                        print("  <<<")
                        print(e.stderr.rstrip())
                        print("  >>>")

                    sample_gfa = None
                    if len(failures) < args.max_report_graphs:
                        sample_gfa = gfa_path.read_text()
                    if args.keep_failing:
                        dest_gfa = Path("/tmp") / f"fail_graph_{i}.gfa"
                        dest_gfa.write_text(gfa_path.read_text())
                        print(f"  GFA saved to {dest_gfa}")

                        dest_stdout = Path("/tmp") / f"fail_graph_{i}_t{t}.bubblefinder.stdout.txt"
                        dest_stderr = Path("/tmp") / f"fail_graph_{i}_t{t}.bubblefinder.stderr.txt"
                        dest_stdout.write_text(e.output or "")
                        dest_stderr.write_text(e.stderr or "")
                        print(f"  BubbleFinder stdout saved to {dest_stdout}")
                        print(f"  BubbleFinder stderr saved to {dest_stderr}")

                        if out_path.exists():
                            dest_sbfind = Path("/tmp") / f"fail_graph_{i}_t{t}.sbfind"
                            dest_sbfind.write_text(out_path.read_text())
                            print(f"  Partial BubbleFinder output (.sbfind) saved to {dest_sbfind}")

                    failures.append({
                        "index": i,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "reasons": sorted(reasons),
                        "threads": threads_list,
                        "gfa": sample_gfa,
                    })
                    n_fail += 1
                    break

            if bf_error:
                continue

            global_divergence = False
            for t in threads_list:
                missing, extra = compare_snarls(snarls_bf, snarls_by_threads[t])
                if missing or extra:
                    global_divergence = True
                    reasons.add("divergence_bruteforce_vs_bubblefinder")
                    print(f"[Graph {i}] DIVERGENCE brute-force vs BubbleFinder (threads={t})")
                    print(f"  nodes={n_nodes}, edges={n_edges}")
                    if missing:
                        print("  Snarls missing in BubbleFinder:")
                        for sn in sorted(missing):
                            print("    ", sn)
                    if extra:
                        print("  Extra snarls in BubbleFinder:")
                        for sn in sorted(extra):
                            print("    ", sn)

            if len(threads_list) > 1:
                ref_t = threads_list[0]
                ref_snarls = snarls_by_threads[ref_t]
                for t in threads_list[1:]:
                    missing, extra = compare_snarls(ref_snarls, snarls_by_threads[t])
                    if missing or extra:
                        global_divergence = True
                        reasons.add("divergence_between_threads")
                        print(f"[Graph {i}] DIVERGENCE between threads={ref_t} and threads={t}")
                        print(f"  nodes={n_nodes}, edges={n_edges}")
                        if missing:
                            print(f"  Snarls missing in threads={t} vs {ref_t}:")
                            for sn in sorted(missing):
                                print("    ", sn)
                        if extra:
                            print(f"  Extra snarls in threads={t} vs {ref_t}:")
                            for sn in sorted(extra):
                                print("    ", sn)

            if global_divergence:
                graph_failed = True

            if graph_failed:
                sample_gfa = None
                if len(failures) < args.max_report_graphs:
                    sample_gfa = gfa_path.read_text()
                if args.keep_failing:
                    dest = Path("/tmp") / f"fail_graph_{i}.gfa"
                    dest.write_text(gfa_path.read_text())
                    print(f"  GFA saved to {dest}")

                failures.append({
                    "index": i,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "reasons": sorted(reasons),
                    "threads": threads_list,
                    "gfa": sample_gfa,
                })
                n_fail += 1
            else:
                print(f"[Graph {i}] OK (nodes={n_nodes}, edges={n_edges}, threads={threads_list})")

    print(f"Total number of graphs tested: {args.n_graphs}")
    print(f"Number of graphs with divergence or errors: {n_fail}")

    if n_fail == 0:
        return

    reason_counts = {}
    for f in failures:
        for r in f["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    if reason_counts:
        print("\nFailure causes summary:")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  - {r}: {c} graphs")

    failed_indices = [f["index"] for f in failures]
    print("\nIndices of failing graphs:")
    print("  ", ", ".join(str(i) for i in failed_indices))

    sample = failures[:args.max_report_graphs]
    print(f"\nDetails for {len(sample)} failing graph(s):")
    for f in sample:
        print("\n------------------------------")
        print(f"Graph {f['index']} (nodes={f['n_nodes']}, edges={f['n_edges']})")
        print(f"Reasons: {', '.join(f['reasons'])}")
        print(f"Thread counts tested: {f['threads']}")
        if f["gfa"] is not None:
            print("GFA:")
            print("<<<")
            print(f["gfa"].rstrip())
            print(">>>")
        else:
            print("GFA not shown (beyond max-report-graphs).")


if __name__ == "__main__":
    main()