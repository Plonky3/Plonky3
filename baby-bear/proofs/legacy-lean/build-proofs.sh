#!/usr/bin/env bash
#
# build-proofs.sh -- regenerate the Lean extraction of `p3-baby-bear` and build it.
#
#   1. apply patches/pre-extraction/*.patch to the RUST SOURCE (none today)
#   2. run `cargo hax into legacy-lean` over the crate
#   3. revert the pre-extraction patches (unconditionally, even on failure)
#   4. snapshot the unmodified output as post-extraction/pristine.snapshot.lean
#   5. apply patches/post-extraction/*.patch (the hand-reconciliation)
#   6. `lake build`
#
# Layout. The Lake package root is this directory; the libraries are split by
# `srcDir` so that hax's output directory holds nothing hax did not write:
#
#   legacy-lean/            <- package root (lakefile, toolchain, manifest)
#     extraction/           <- hax's default output dir; GENERATED + the
#                              axiomatized interface needed to typecheck it
#     spec/                 <- what we assert: p3_baby_bear_proofs/
#     patches/
#       pre-extraction/     <- patches to Rust source, applied BEFORE hax
#       post-extraction/    <- patches to generated Lean, applied AFTER hax
#       check-patches.sh    <- conventions gate, run before anything is applied
#       new-patch.sh        <- author a new patch
#     README.md  TCB.md  SYNC.md
#
# Patches are hand-owned and each carries its own rationale in its header; there
# is no README cataloguing them. `check-patches.sh` enforces that.
#
# `pre-extraction/` is EMPTY, and that emptiness is a claim: no Rust source is
# modified to make this extraction work. The SIMD backends are excluded by
# extracting for a target that has none (see HAX_TARGET below) rather than by
# patching out `cfg`s, so there is nothing to patch and nothing to revert, and a
# normal `cargo build` is completely unaffected. If a pre-extraction patch is
# ever added, it changes the artifact under verification -- a strictly worse
# trust position than a post-extraction patch. See patches/README.md and TCB.md.
#
# Usage:  ./baby-bear/proofs/legacy-lean/build-proofs.sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"             # baby-bear/
PKG_DIR="$SCRIPT_DIR"                                     # Lake package root
# hax writes to <crate>/proofs/<backend>/extraction/ and that is not
# configurable, so this path is fixed by the tool, not by us.
EXTRACT_DIR="$PKG_DIR/extraction"
PRE_PATCH_DIR="$PKG_DIR/patches/pre-extraction"
POST_PATCH_DIR="$PKG_DIR/patches/post-extraction"
TARGET="$EXTRACT_DIR/p3_baby_bear.lean"
PRISTINE="$POST_PATCH_DIR/pristine.snapshot.lean"
# `git apply` paths in pre-extraction patches are repo-root relative (-p1).
REPO_ROOT="$(cd "$CRATE_ROOT/.." && pwd)"

# --- tunables -----------------------------------------------------------------
# hax's pinned nightly. Needed so `-Z` flags are accepted even by cargo's
# metadata probe.
HAX_TOOLCHAIN="${HAX_TOOLCHAIN:-nightly-2025-11-08}"
# The Rust driver spawns the OCaml engine and aborts if their revisions differ,
# so this must be the engine built from the same hax revision as `cargo-hax`.
HAX_ENGINE_BINARY="${HAX_ENGINE_BINARY:-$HOME/.opam/hax/bin/hax-engine}"
# A target with no SIMD. `baby-bear/src/lib.rs` gates aarch64_neon / x86_64_avx2
# / x86_64_avx512 on target_arch+target_feature, and `monty-31/src/lib.rs`
# selects its `no_packing` path under the same negation, so a bare-metal triple
# excludes every SIMD module across the whole dependency graph. This is the
# configuration CI's `check_embedded` job already builds.
HAX_TARGET="${HAX_TARGET:-thumbv7em-none-eabi}"
# `p3-util` calls `<[MaybeUninit<T>]>::assume_init_ref`, still feature-gated on
# the pinned nightly. hax appends its own `--cfg hax` to whatever is set here.
HAX_RUSTFLAGS="${HAX_RUSTFLAGS:--Zcrate-attr=feature(maybe_uninit_slice)}"
# Isolated target dir: a warm shared target/hax serves stale Lean.
HAX_TARGET_DIR="${HAX_TARGET_DIR:-/tmp/hax-extract/target-baby-bear-legacy-lean}"

if [ ! -x "$HAX_ENGINE_BINARY" ]; then
    echo "error: hax engine not found at $HAX_ENGINE_BINARY" >&2
    echo "       Set HAX_ENGINE_BINARY to the engine matching your cargo-hax." >&2
    exit 1
fi

# --- the patch applier, shared by both phases ---------------------------------
# `patch -F0` forbids fuzz: context drift fails loudly instead of applying
# somewhere merely plausible. Line-number offsets are still tolerated, which is
# what lets the patches in a phase shift each other harmlessly.
PATCH_FLAGS="-p1 -F0 --no-backup-if-mismatch"

# Two passes: dry-run the whole ordered set, then apply. With several patches per
# phase this restores the all-or-nothing behaviour a single patch had for free.
# $1 = phase dir, $2 = root to apply in, $3 = file recording what was applied
apply_phase() {
    local dir="$1" root="$2" record="$3"
    local pf n=0
    for pf in "$dir"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        if ! patch $PATCH_FLAGS --dry-run -d "$root" < "$pf" >/dev/null 2>&1; then
            echo "error: $(basename "$pf") does not apply cleanly." >&2
            echo "       Nothing has been changed. Re-run with --dry-run removed" >&2
            echo "       to see the rejects, or fix that patch." >&2
            return 1
        fi
        n=$((n + 1))
    done
    if [ "$n" -eq 0 ]; then
        echo "    none"
        return 0
    fi
    for pf in "$dir"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        patch $PATCH_FLAGS -d "$root" < "$pf" >/dev/null
        [ -z "$record" ] || echo "$pf" >> "$record"
        printf '    %-46s %s\n' "$(basename "$pf")" "[$(sed -n 's/^# Cost:[[:space:]]*//p' "$pf" | head -1)]"
    done
}

# Reverted on EXIT so an aborted run never leaves the Rust tree modified. We
# reverse exactly what we applied, in reverse order -- never `git checkout`,
# which would discard unrelated uncommitted work. The record is a file rather
# than a bash array because macOS still ships bash 3.2.
PRE_RECORD="$(mktemp)"
revert_pre_patches() {
    [ -s "$PRE_RECORD" ] || return 0
    local ok=1 pf
    # tail -r reverses on BSD; tac on GNU.
    for pf in $( (tail -r "$PRE_RECORD" 2>/dev/null || tac "$PRE_RECORD") ); do
        patch $PATCH_FLAGS -R -d "$REPO_ROOT" < "$pf" >/dev/null || ok=0
    done
    : > "$PRE_RECORD"
    if [ "$ok" = "1" ]; then
        echo "    reverted pre-extraction patches"
    else
        echo "ERROR: could not revert pre-extraction patches. The Rust tree is" >&2
        echo "       left MODIFIED -- inspect 'git status' before building." >&2
    fi
}
cleanup() { revert_pre_patches; rm -f "$PRE_RECORD"; }
trap cleanup EXIT

echo "==> 0/6 checking patch conventions"
"$PKG_DIR/patches/check-patches.sh"

echo "==> 1/6 pre-extraction patches (Rust source)"
if ls "$PRE_PATCH_DIR"/[0-9]*.patch >/dev/null 2>&1; then
    echo "    WARNING: patching Rust source changes the artifact under" >&2
    echo "             verification. See TCB.md layer 4 before adding these." >&2
fi
apply_phase "$PRE_PATCH_DIR" "$REPO_ROOT" "$PRE_RECORD"

echo "==> 2/6 extracting with hax (backend: legacy-lean, target: $HAX_TARGET)"
cd "$CRATE_ROOT"
rm -rf "$HAX_TARGET_DIR"
# hax can print errors and still write a usable file; judge the result from
# `lake build`, not from hax's exit status. But if no file appeared, stop.
env RUSTUP_TOOLCHAIN="$HAX_TOOLCHAIN" \
    HAX_ENGINE_BINARY="$HAX_ENGINE_BINARY" \
    RUSTFLAGS="$HAX_RUSTFLAGS" \
    CARGO_TARGET_DIR="$HAX_TARGET_DIR" \
    cargo hax -C --target "$HAX_TARGET" ';' into legacy-lean \
  || echo "WARNING: cargo hax exited non-zero; inspecting output anyway." >&2

if [ ! -f "$TARGET" ]; then
    echo "error: hax produced no $TARGET" >&2
    exit 1
fi

# Revert now, before the Lean build, so the rest of the run sees a clean tree.
# The EXIT trap stays armed and becomes a no-op.
echo "==> 3/6 restoring the Rust source"
if [ -s "$PRE_RECORD" ]; then
    revert_pre_patches
else
    echo "    nothing to revert (no pre-extraction patches)"
fi

echo "==> 4/6 snapshotting pristine output"
mkdir -p "$POST_PATCH_DIR"
cp "$TARGET" "$PRISTINE"

echo "==> 5/6 applying post-extraction patches (generated Lean)"
apply_phase "$POST_PATCH_DIR" "$PKG_DIR" ""

echo "==> 6/6 lake build"
# CompPoly pulls mathlib into the dependency graph; without the prebuilt cache
# this becomes an hours-long from-source build. `cache get` is a no-op once warm.
(cd "$PKG_DIR" && lake exe cache get >/dev/null 2>&1) || \
    echo "note: 'lake exe cache get' failed; mathlib may build from source" >&2
if (cd "$PKG_DIR" && lake build); then
    n_sorry=$(grep -c 'sorry' "$TARGET" || true)
    echo "    extraction contains $n_sorry 'sorry' occurrences"
else
    echo "WARNING: lake build failed. The patched file is in place; inspect for" >&2
    echo "         drift. Fix the failing patch (its .rej shows what moved), or" >&2
    echo "         author a new one with patches/new-patch.sh." >&2
    exit 1
fi

if [ -t 1 ]; then
    green=$'\033[32m'; yellow=$'\033[33m'; reset=$'\033[0m'
else
    green=; yellow=; reset=
fi
echo "${green}Done. Build passed. ✓${reset}"
echo "${yellow}A green build means the extraction TYPE-CHECKS, meaning that"
echo "the theorems in spec/ hold up to the assumptions; See TCB.md.${reset}"
