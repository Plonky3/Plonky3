#!/usr/bin/env python3
"""test-pre-patches.py -- run upstream's own tests with and without the
pre-extraction patches, and make every difference something a patch declares.

A pre-extraction patch changes the Rust that is extracted, so "it still
compiles" is not evidence that it changes nothing else. This script builds two
copies of the workspace, `base` (the tree as it is) and `patched` (with
patches/pre-extraction/*.patch applied in order), runs
`cargo test --workspace --no-fail-fast` in each, and compares the results per
test.

A *divergence* is any of:
  - a test that passes in `base` but fails, or is missing, in `patched`;
  - `patched` failing to build its tests at all (reported as `(build)`);
  - an upstream test whose source differs between the two trees. The tests are
    inline, so a patch could weaken a test instead of the code. Every
    `#[test]` fn's source is compared, found by brace matching (not a full
    Rust parser).

Each divergence is attributed to the first patch that introduces it, by
re-running on prefixes of the patch list (only when there is one), and the
script **writes it into that patch's header**, in a block it maintains:

    # Tested:     test-pre-patches.py: 4625 of 4671 upstream tests pass as
    #             shipped, 4625 with the pre-patches; 1 divergence from this patch.
    # Divergence: p3_mds[unittests src/lib.rs]::util::tests::some_test
    #   observed: passes as shipped; FAILED with the patches: assertion failed ..
    #   class:    TODO
    #   was:      TODO
    #   now:      TODO

`Tested:` and `observed:` are the script's; it rewrites them every run.
`class`, `was`, `now` (and optionally `witness`, `approved`) are the author's
and are kept across runs: `class` is one of

    totality | observable | signature | test-edited | harness

and `was`/`now` say, in words, what the patch changed. Entries that no longer
diverge are removed. The run fails while any entry is unclassified (`TODO`),
so the build stops until a person has judged each divergence.

`--check` does not write: it fails if a header is out of date instead (for CI).

Results are cached on the tree state, the *diff* part of each patch (header
edits do not invalidate it) and this script, so re-runs are free.

Usage: test-pre-patches.py [--no-cache] [--check]      (run from anywhere)
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[2]
PATCH_DIR = HERE / "pre-extraction"
WORK = Path(os.environ.get("PRETEST_DIR", "/tmp/hax-extract/pretest"))
CLASSES = {"totality", "observable", "signature", "test-edited", "harness"}
# The proofs directories hold Lean, not Rust, and are large; the tests never see them.
SYNC_EXCLUDES = ["target/", ".git/", "**/.lake/", "baby-bear/proofs/"]


def sh(cmd, cwd=None, env=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if check and r.returncode != 0:
        sys.exit(f"error: {' '.join(cmd)} failed:\n{r.stdout}{r.stderr}")
    return r


def patches():
    return sorted(PATCH_DIR.glob("[0-9]*.patch"))


DIFF_START = re.compile(r"^(diff --git |--- )", re.M)


def split_header(text):
    """(header, diff) of a patch file; the header is everything before the
    first `diff --git` / `--- ` line."""
    m = DIFF_START.search(text)
    return (text, "") if m is None else (text[:m.start()], text[m.start():])


def cache_key(ps):
    h = hashlib.sha256()
    # The tree's content, not the commit: a commit that only touches the
    # proofs directory (which the tests never see) must not invalidate this.
    tree = sh(["git", "ls-tree", "-r", "HEAD"], cwd=REPO).stdout.splitlines()
    h.update("\n".join(l for l in tree if "\tbaby-bear/proofs/" not in l).encode())
    # Uncommitted changes to tracked Rust/manifests are part of the tree tested.
    h.update(sh(["git", "diff", "HEAD", "--", ".", ":(exclude)baby-bear/proofs"], cwd=REPO).stdout.encode())
    for p in ps:
        h.update(p.name.encode())
        # Only the diff: the header is rewritten by this script and by authors,
        # and neither changes what the patch does.
        h.update(split_header(p.read_text())[1].encode())
    h.update(Path(__file__).read_bytes())
    h.update(sh(["rustc", "--version"], cwd=REPO).stdout.encode())
    return h.hexdigest()[:16]


def sync(dest):
    """Mirror the repo's tracked and unignored files into `dest`, preserving
    mtimes so that cargo's incremental cache in the per-tree target dir stays
    valid between runs."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--delete", "--out-format=%n"]
    for e in SYNC_EXCLUDES:
        cmd += ["--exclude", e]
    out = sh(cmd + [f"{REPO}/", f"{dest}/"]).stdout
    # `-a` restores the repo's mtime on every file it rewrites. When a previous
    # run left a *patched* copy here, that mtime is older than the artifact
    # cargo built from the patched copy, so cargo would rerun a stale binary.
    # Give every rewritten file a fresh mtime so cargo rebuilds it.
    for name in out.splitlines():
        f = dest / name
        if f.is_file():
            os.utime(f)


def apply(tree, ps):
    for p in ps:
        r = subprocess.run(["patch", "-p1", "-F0", "--no-backup-if-mismatch", "-s", "-d", str(tree)],
                           stdin=open(p), text=True, capture_output=True)
        if r.returncode != 0:
            sys.exit(f"error: {p.name} does not apply to the test tree:\n{r.stdout}{r.stderr}")


RUNNING = re.compile(r"^\s+Running (\S+(?: \S+)?) \((?:[^)]*/)?([A-Za-z0-9_-]+?)(?:-[0-9a-f]{16})?\)$")
DOCTESTS = re.compile(r"^\s+Doc-tests (\S+)$")
RESULT = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)(?:, .*)?$")
DOC_LINE = re.compile(r" \(line \d+\)$")
SECTION = re.compile(r"^---- (.+?) stdout ----$")
PANIC = re.compile(r"^thread '.*'(?: \(\d+\))? panicked at ")


def run_tests(tree, target):
    """`cargo test` over the workspace. Returns ({test: status}, build_ok, log)."""
    # No debuginfo and no incremental cache: the tests do not need them, and
    # they are most of a target dir's size (~9 GB -> a few GB per tree).
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_TERM_COLOR="never",
               CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0",
               CARGO_INCREMENTAL="0")
    # One stream: cargo prints `Running <binary>` on stderr and the test results
    # on stdout, and the parser relies on their interleaving.
    r = subprocess.run(["cargo", "test", "--workspace", "--no-fail-fast", "--locked"],
                       cwd=tree, env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    log = r.stdout
    results, messages, binary, section = {}, {}, "?", None
    for line in log.splitlines():
        m = RUNNING.match(line)
        if m:
            binary, section = f"{m.group(2)}[{m.group(1)}]", None
            continue
        m = DOCTESTS.match(line)
        if m:
            binary, section = f"{m.group(1)}[doc]", None
            continue
        m = RESULT.match(line)
        if m:
            # A patch above a doctest shifts its line number; that is not a change.
            name = DOC_LINE.sub("", m.group(1)) if binary.endswith("[doc]") else m.group(1)
            key = f"{binary}::{name}"
            # Several doctests can share a name once line numbers are dropped.
            n = 1
            while key in results:
                n += 1
                key = f"{binary}::{name}#{n}"
            results[key] = m.group(2)
            continue
        # A failing test's captured output: `---- <name> stdout ----`, then
        # `thread '..' panicked at <loc>:` and the message on the next line.
        m = SECTION.match(line)
        if m:
            section = f"{binary}::{DOC_LINE.sub('', m.group(1))}"
            continue
        if section and section not in messages:
            if PANIC.match(line):
                messages[section] = ""
            continue
        if section and messages.get(section) == "" and line.strip():
            messages[section] = line.strip()
    build_ok = "error: could not compile" not in log and "error[E" not in log
    if not build_ok:
        messages["(build)"] = next((l.strip() for l in log.splitlines()
                                    if l.startswith(("error[", "error: "))), "")
    return results, build_ok, log, messages


TEST_ATTR = re.compile(r"#\[(?:test|tokio::test|proptest)\b[^\]]*\]")
FN_NAME = re.compile(r"\bfn\s+([A-Za-z0-9_]+)")


def test_sources(tree):
    """{(file, fn): source} for every `#[test]`-annotated fn, by brace matching."""
    out = {}
    for f in tree.rglob("*.rs"):
        rel = f.relative_to(tree)
        if rel.parts[0] in ("target",) or ".lake" in rel.parts:
            continue
        s = f.read_text(errors="replace")
        for m in TEST_ATTR.finditer(s):
            fm = FN_NAME.search(s, m.end())
            if not fm:
                continue
            i = s.find("{", fm.end())
            if i < 0:
                continue
            depth, j = 0, i
            while j < len(s):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out[(str(rel), fm.group(1))] = s[m.start():j + 1]
    return out


def diverge(base, patched, base_src, patched_src):
    """{test: what was observed}, for every divergence."""
    def why(t):
        msg = patched[3].get(t, "")
        return f": {msg}" if msg else ""
    div = {}
    if not patched[1]:
        div["(build)"] = f"the tree as shipped builds its tests; with the patches it does not{why('(build)')}"
    for t, st in base[0].items():
        if st == "ok":
            pst = patched[0].get(t)
            if pst is None:
                div[t] = "passes as shipped; missing with the patches (not built, or not run)"
            elif pst != "ok":
                div[t] = f"passes as shipped; {pst} with the patches{why(t)}"
    for k, src in base_src.items():
        p = patched_src.get(k)
        if p is None:
            div[f"{k[0]}::{k[1]}"] = "test source removed by the patches"
        elif p != src:
            div[f"{k[0]}::{k[1]}"] = "test source modified by the patches"
    return div


AUTHOR_FIELDS = ("class", "was", "now", "witness", "approved")
FIELD = re.compile(r"^#\s+(observed|class|witness|was|now|approved):\s*(.*)$")


def declarations(p):
    """{test: {field: value}} from the `# Divergence:` entries of a header."""
    decl, cur = {}, None
    for line in split_header(p.read_text())[0].splitlines():
        m = re.match(r"^# Divergence:\s*(.+?)\s*$", line)
        if m:
            cur = m.group(1)
            decl[cur] = {}
            continue
        m = FIELD.match(line)
        if m and cur:
            decl[cur][m.group(1)] = m.group(2).strip()
        elif not line.startswith("#   ") and not line.startswith("#             "):
            cur = None
    return decl


def one_line(s, n=160):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def render_block(res, name, mine, old):
    """The script-owned part of a patch header."""
    n = len(mine)
    out = [f"# Tested:     test-pre-patches.py: {res['passed_base']} of {res['tests_base']} upstream tests pass as",
           f"#             shipped, {res['passed_patched']} with the pre-patches; "
           f"{n} divergence{'' if n == 1 else 's'} from this patch."]
    for t in sorted(mine):
        prev = old.get(t, {})
        out.append(f"# Divergence: {t}")
        out.append(f"#   observed: {one_line(mine[t])}")
        for f in AUTHOR_FIELDS:
            if f in prev:
                out.append(f"#   {f + ':':<9} {prev[f]}")
            elif f in ("class", "was", "now"):
                out.append(f"#   {f + ':':<9} TODO")
    return out


def rewrite_header(p, block):
    """Replace the `# Tested:`/`# Divergence:` lines of p's header with `block`
    (placed just before the diff), leaving every other header line alone."""
    header, diff = split_header(p.read_text())
    lines = header.rstrip("\n").split("\n")
    kept, skipping = [], False
    for line in lines:
        if line.startswith("# Tested:") or line.startswith("# Divergence:"):
            skipping = True
            continue
        if skipping and (line.startswith("#   ") or line.startswith("#             ")):
            continue
        skipping = False
        kept.append(line)
    while kept and kept[-1] == "#":
        kept.pop()
    new = "\n".join(kept + ["#"] + block + ["#"]) + "\n" + diff
    return new


def evaluate(ps, tag):
    """Test `base` and `patched` (with patch list `ps`). `tag` names the tree."""
    # One patched tree and target dir, reused for every prefix: cargo only
    # rebuilds what the patch set changes, and disk use stays at two trees.
    base_tree, pat_tree = WORK / "base", WORK / "patched"
    sync(base_tree)
    sync(pat_tree)
    apply(pat_tree, ps)
    base = run_tests(base_tree, WORK / "target-base")
    if not base[1]:
        sys.exit("error: the tree as shipped does not build its tests; nothing to compare.\n"
                 + base[2][-3000:])
    pat = run_tests(pat_tree, WORK / "target-patched")
    (WORK / "log-base.txt").write_text(base[2])
    (WORK / f"log-patched-{tag}.txt").write_text(pat[2])
    return base, pat, diverge(base, pat, test_sources(base_tree), test_sources(pat_tree))


def main():
    ps = patches()
    if not ps:
        print("    no pre-extraction patches; nothing to test")
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    key = cache_key(ps)
    cache = WORK / f"result-{key}.json"
    if cache.exists() and "--no-cache" not in sys.argv:
        res = json.loads(cache.read_text())
        print(f"    cached ({key})")
    else:
        base, pat, div = evaluate(ps, "all")
        print(f"    full patch set: {len(div)} divergence(s) before attribution "
              f"(logs in {WORK})")
        for t, w in sorted(div.items()):
            print(f"      {t}: {w}")
        attributed = {}
        if div and len(ps) > 1:
            # First prefix of the patch list that shows each divergence.
            remaining = set(div)
            for k in range(1, len(ps) + 1):
                _, _, d = evaluate(ps[:k], f"prefix{k}")
                for t in list(remaining):
                    if t in d:
                        attributed[t] = ps[k - 1].name
                        remaining.discard(t)
                if not remaining:
                    break
        elif div:
            attributed = {t: ps[0].name for t in div}
        n_base = sum(1 for s in base[0].values() if s == "ok")
        n_pat = sum(1 for s in pat[0].values() if s == "ok")
        added = sorted(set(pat[0]) - set(base[0]))
        res = {"key": key, "tests_base": len(base[0]), "passed_base": n_base,
               "passed_patched": n_pat, "failed_base": sorted(t for t, s in base[0].items() if s == "FAILED"),
               "added": added,
               "divergences": {t: {"what": w, "patch": attributed.get(t, "?")} for t, w in div.items()}}
        cache.write_text(json.dumps(res, indent=1))

    print(f"    upstream tests: {res['passed_base']} passed as shipped, "
          f"{res['passed_patched']} passed with the pre-patches "
          f"({res['tests_base']} run)")
    if res["failed_base"]:
        print(f"    note: {len(res['failed_base'])} test(s) already fail as shipped (not attributed):")
        for t in res["failed_base"]:
            print(f"      {t}")

    fail = False
    unattributed = [t for t, d in res["divergences"].items() if d["patch"] not in {p.name for p in ps}]
    for t in unattributed:
        print(f"    ERROR: divergence {t} appears with the full set but with no prefix of it "
              f"(flaky test?): {res['divergences'][t]['what']}")
        fail = True

    check_only = "--check" in sys.argv
    for p in ps:
        old = declarations(p)
        mine = {t: d["what"] for t, d in res["divergences"].items() if d["patch"] == p.name}
        new = rewrite_header(p, render_block(res, p.name, mine, old))
        if new != p.read_text():
            if check_only:
                print(f"    OUT OF DATE: {p.name}'s header does not match this run; "
                      f"re-run without --check to rewrite it")
                fail = True
            else:
                p.write_text(new)
                print(f"    wrote {len(mine)} divergence(s) into {p.name}")
        for t in sorted(mine):
            cls = old.get(t, {}).get("class", "TODO")
            if cls not in CLASSES:
                print(f"    UNCLASSIFIED divergence from {p.name}: {t}")
                print(f"      {one_line(mine[t])}")
                fail = True
            else:
                print(f"    declared divergence ({cls}) from {p.name}: {t}")
    if fail and any(d["patch"] != "?" for d in res["divergences"].values()):
        print("    Classify each entry in the patch header (class/was/now; see patches/README.md),")
        print("    or fix the patch so the test no longer diverges, and re-run.")
    if not res["divergences"]:
        print("    no divergences")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
