#!/usr/bin/env python3
"""Build the local public Rust API and complete C/C++ header reference."""

import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def build_reference(source, output, toolchain="1.93.0"):
    source = Path(source).resolve()
    output = Path(output).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    cargo = (source / "Cargo.toml").read_text()
    version = re.search(r'^version\s*=\s*"([^"]+)"', cargo, re.M).group(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".spqr-reference-", dir=output.parent))
    try:
        subprocess.run(
            ["rustup", "run", toolchain, "rustdoc", "--edition", "2021",
             "--crate-name", "spqr_rust", "--crate-type", "rlib",
             "--crate-version", version, str(source / "src/lib.rs"),
             "-o", str(staging)],
            check=True,
        )
        (staging / "index.html").write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="0;url=spqr_rust/index.html">'
            '<title>SPQR-Rust API</title></head><body>'
            '<p><a href="spqr_rust/index.html">SPQR-Rust API reference</a></p>'
            '</body></html>'
        )
        headers = staging / "headers"
        headers.mkdir()
        inventory = {"commit": commit, "version": version, "toolchain": toolchain,
                     "generated": datetime.date.today().isoformat(), "headers": {}}
        for path in sorted((source / "include").iterdir()):
            if not path.is_file():
                continue
            text = path.read_text()
            shutil.copyfile(path, headers / path.name)
            declarations = []
            pattern = r"^[ \t]*(?:[\w*]+[ \t*]+)+(sp(?:qr|_compress)_[A-Za-z0-9_]+)\s*\([^;{}]*\)\s*;"
            for match in re.finditer(pattern, text, re.M):
                line = text.count("\n", 0, match.start()) + 1
                declarations.append({"name": match.group(1), "line": line,
                                     "signature": " ".join(match.group(0).split())})
            types = []
            for match in re.finditer(r"^[ \t]*(?:class|struct)\s+([A-Za-z_]\w*)[^;\n]*\{", text, re.M):
                types.append({"name": match.group(1),
                              "line": text.count("\n", 0, match.start()) + 1})
            inventory["headers"][path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "functions": declarations, "types": types,
            }
            function_links = " · ".join(
                f'<a href="#L{item["line"]}">{html.escape(item["name"])}</a>'
                for item in declarations
            )
            type_links = " · ".join(
                f'<a href="#L{item["line"]}">{html.escape(item["name"])}</a>'
                for item in types
            )
            lines = "\n".join(
                f'<span id="L{i}"><a class="source-line" href="#L{i}">{i:4}</a> {html.escape(line)}</span>'
                for i, line in enumerate(text.splitlines(), 1)
            )
            name = html.escape(path.name)
            page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name} · SPQR-Rust header reference</title>
<link rel="stylesheet" href="../../assets/styles.css?v=6"></head>
<body><div class="site-wrap">
<header class="site-header"><a class="brand" href="../../spqr-rust.html">SPQR-Rust</a></header>
<nav class="main-nav" aria-label="Main navigation"><ul>
<li><a href="../../spqr-rust-documentation.html">Documentation</a></li>
<li><a href="../../spqr-rust-api.html">API reference</a></li></ul></nav>
<main class="header-reference"><h1>{name}</h1>
<p><a href="../../spqr-rust-api.html#headers">Header index</a> · <a href="{name}" download>Download header</a></p>
<p class="small-note">SPQR-Rust {version}, source <a href="https://github.com/algbio/spqr-rust/tree/{commit}">{commit[:8]}</a>. Complete header, including declarations, definitions and comments.</p>
<details><summary>Jump to a declaration</summary><p>{function_links or type_links or "See the source below."}</p></details>
<pre class="header-source"><code>{lines}</code></pre></main>
<footer class="site-footer">SPQR-Rust · MIT · Generated source reference</footer>
</div></body></html>'''
            (headers / (path.name + ".html")).write_text(page)
        (staging / "reference.json").write_text(json.dumps(inventory, indent=2) + "\n")
        (staging / "NOTICE.txt").write_text(
            f"SPQR-Rust {version}, commit {commit}\n"
            "https://github.com/algbio/spqr-rust\n"
            "The library declares the MIT license. Rust API pages were generated by rustdoc; "
            "its bundled static resources retain their supplied license files.\n"
        )
        if output.exists():
            raise FileExistsError(f"Use a fresh output directory, then review and replace: {output}")
        staging.rename(output)
        return inventory
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local SPQR-Rust checkout")
    parser.add_argument("--output", required=True, help="New destination directory")
    parser.add_argument("--toolchain", default="1.93.0")
    args = parser.parse_args()
    metadata = build_reference(args.source, args.output, args.toolchain)
    print(f"Built SPQR-Rust {metadata['version']} reference at {args.output}")
