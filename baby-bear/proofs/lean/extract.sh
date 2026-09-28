#!/usr/bin/env bash
#
# extract.sh -- build the Lean extraction of `p3-baby-bear` from a fresh
# checkout: check the pinned tools, regenerate the Lean from the Rust with
# hax's current Lean backend (charon + aeneas), and check the proofs.
#
# The generated Lean is not committed (generated/ is gitignored), so this script
# is the way to build. After one run, `lake build` alone rebuilds the proofs.
#
#   0. tools: check that `cargo-hax` is the pinned release; let it fetch its
#      charon and aeneas; install the Rust toolchain charon needs; check that
#      lakefile.toml, lean-toolchain and SYNC.md's pin table agree
#   1. check the patch conventions (patches/check-patches.sh)
#   2. run upstream's tests with and without the pre-extraction patches, and
#      record every difference in the patch headers (patches/test-pre-patches.py)
#   3. apply patches/pre-extraction/*.patch to the RUST SOURCE, and
#      `cargo check` the variant charon extracts (the pre-patches' safety net)
#   4. run `cargo hax into lean` over the crate, and over each dependency
#      crate scoped to exactly the items p3-baby-bear uses (see DEPS below)
#   5. revert the pre-extraction patches (unconditionally, even on failure)
#   6. check the hand-written stubs in assumptions/ against the templates
#      aeneas regenerated, then snapshot the output as generated/pristine/
#   7. apply patches/post-extraction/*.patch (the hand-reconciliation)
#   8. `lake build`
#
# Layout (see README.md):
#
#   lean/                  <- Lake package root (lakefile, toolchain, manifest)
#     generated/           <- MACHINE OUTPUT, gitignored, rewritten every run
#       p3-baby-bear/        hax's --output-dir for the crate
#       p3-monty-31/ p3-mds/ the same, for each scoped dependency extraction
#       pristine/            the output before post-extraction patches
#     assumptions/         <- hand-written and trusted (TCB.md, layer 3)
#     spec/                <- hand-written theorems
#     patches/             <- hand-written diffs, to the Rust and to generated/
#
# `patches/pre-extraction/` is NOT empty. The Lean
# backend cannot translate the generic associated types that rustc desugars
# `-> impl Iterator` trait methods into, and p3-field is full of them. The
# patches hide exactly those items behind `cfg(hax_backend_lean)` and change
# nothing else. Step 2 shows the normal build is unaffected (upstream's own
# tests, per test), and step 3 has rustc confirm that nothing still reachable
# from p3-baby-bear depended on the hidden items. Each patch header gives its
# own argument; see patches/README.md and TCB.md.
#
# Usage:  ./baby-bear/proofs/lean/extract.sh [--tools-only]
#
# Environment:
#   HAX_BIN     the cargo-hax to use (default: `cargo-hax` on PATH). It must be
#               the pinned release: `cargo install --locked cargo-hax@<HAX_VERSION>`
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"             # baby-bear/
PKG_DIR="$SCRIPT_DIR"                                     # Lake package root
GEN_DIR="$PKG_DIR/generated"
ASSUME_DIR="$PKG_DIR/assumptions"
PRE_PATCH_DIR="$PKG_DIR/patches/pre-extraction"
POST_PATCH_DIR="$PKG_DIR/patches/post-extraction"
PRISTINE="$GEN_DIR/pristine"
# Pre-extraction patch paths are repo-root relative (-p1).
REPO_ROOT="$(cd "$CRATE_ROOT/.." && pwd)"

# --- pins ---------------------------------------------------------------------
# Keep these in step with the "Current pins" table in SYNC.md; step 0 fails
# otherwise. hax >= 0.4 needs no engine: `into lean` downloads and runs charon
# and aeneas, checksum-verified. The versions it uses are the built-in defaults
# of the hax release, so pinning hax pins them; step 0 still checks each one,
# because a `hax.toml` anywhere above the crate would override them.
HAX_VERSION="0.4.1"
# What tag cargo-hax-v$HAX_VERSION points to. A crates.io build reports
# `version=$HAX_VERSION`; a build from git at the tag reports this commit.
HAX_COMMIT="8d6a41802f32918fe7ebbaad0dfc86937ba3011d"
EXPECT_CHARON="nightly-2026.09.02"
EXPECT_AENEAS="build-183e4f0"
# charon's driver links against rustc, so it runs under this exact nightly
# (from charon's own `rust-toolchain`). The same nightly checks the patched
# tree in step 3, so that both see the same compiler.
CHARON_TOOLCHAIN="nightly-2026-08-18"
CHARON_COMPONENTS="rustc-dev,llvm-tools,rust-src"

# --- extraction settings ------------------------------------------------------
# A target with no SIMD: `baby-bear/src/lib.rs` and
# `monty-31/src/lib.rs` gate every SIMD backend on target_arch/target_feature,
# so a bare-metal triple drops all of them. The hax driver does NOT pass
# `-C --target` on to charon; charon's own `--targets` is what reaches it.
# charon's toolchain file does not include this target, so step 0 adds it.
HAX_TARGET="thumbv7em-none-eabi"
# Items kept out of the translation, for every run. These restrict *scope*:
# nothing here edits code that is translated. The TCB counts each one.
#   Debug impls   -> opaque: the impls stay (`MontyParameters: Debug` needs the
#                    witness) but the `core::fmt` bodies, which aeneas cannot
#                    model, become assumptions.
#   serde         -> excluded: `Field: Serialize + DeserializeOwned` pulls in
#                    serde's mutually recursive `Serializer` trait family.
# NB: charon splits option values on ',', so patterns cannot contain one.
COMMON_CHARON_ARGS="--targets $HAX_TARGET \
--opaque '{impl core::fmt::Debug for _}' \
--exclude serde_core --exclude serde \
--exclude '{impl serde_core::ser::Serialize for _}' \
--exclude '{impl serde_core::de::Deserialize for _}'"
# Scoped dependency extractions. aeneas only ever emits the crate it is run on,
# so the dependency items p3-baby-bear's output refers to have to come from
# somewhere: these runs translate exactly those items, rooted with
# `--start-from` (which also suppresses the default `crate` root). What they
# cannot translate is hand-written in assumptions/Interface/. Whole-crate
# extraction of either crate fails (p3-field's trait cycle; see TCB.md). One
# line per crate:
#   <crate dir>|<output dir under generated/>|<charon roots>
M31=p3_monty_31
DEPS="monty-31|p3-monty-31|\
--start-from $M31::data_traits::MontyParameters \
--start-from $M31::data_traits::PackedMontyParameters \
--start-from $M31::data_traits::BarrettParameters \
--start-from $M31::data_traits::FieldParameters \
--start-from $M31::data_traits::RelativelyPrimePower \
--start-from $M31::data_traits::TwoAdicData \
--start-from $M31::data_traits::BinomialExtensionData \
--start-from $M31::mds::MDSUtils \
--start-from $M31::poseidon1::PartialRoundBaseParameters \
--start-from $M31::poseidon1::PartialRoundParameters \
--start-from $M31::poseidon2::InternalLayerBaseParameters \
--start-from $M31::poseidon2::InternalLayerParameters \
--start-from $M31::monty_31::MontyField31::new \
--start-from '{impl core::clone::Clone for $M31::monty_31::MontyField31<_>}'
mds|p3-mds|--start-from p3_mds::util::first_row_to_first_col"
# Every generated Lean library: <output dir under generated/>|<Lean library>.
LIBS="p3-baby-bear|P3BabyBear p3-monty-31|P3Monty31 p3-mds|P3Mds"
# The pre-extraction patches gate their changes on `hax_backend_lean`, the cfg
# hax's Lean backend defines. hax passes it (as `--rustc-arg`) only to the crate
# being extracted, but the gated items live in its dependencies (p3-field,
# p3-matrix, …), so it goes in RUSTFLAGS too; hax prepends RUSTFLAGS to its own
# flags for every crate. Normal builds never set it, which is why upstream's
# tests (step 2) see the crates exactly as shipped.
EXTRACT_RUSTFLAGS="--cfg hax_backend_lean"
# Isolated target dir for the step-3 check, so it cannot poison a normal build.
CHECK_TARGET_DIR="${CHECK_TARGET_DIR:-/tmp/hax-extract/target-baby-bear-lean-check}"

TOOLS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --tools-only) TOOLS_ONLY=1 ;;
        -h|--help) sed -n '2,/^set -eu/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

# --- 0. tools -----------------------------------------------------------------
echo "==> 0/8 tools"
need() {   # need <command> <how to get it>
    command -v "$1" >/dev/null 2>&1 && return 0
    echo "error: '$1' not found. $2" >&2
    exit 1
}
need rustup  "Install Rust with rustup: https://rustup.rs"
need cargo   "Install Rust with rustup: https://rustup.rs"
need lake    "Install Lean with elan: https://github.com/leanprover/elan (lean-toolchain pins the version)"
need python3 "Install Python 3."
need rsync   "Install rsync."
need patch   "Install patch."
need git     "Install git."

# The toolchain charon runs under, with the target it does not ship.
# Idempotent: a no-op once installed.
if ! rustup target list --toolchain "$CHARON_TOOLCHAIN" --installed 2>/dev/null \
        | grep -qx "$HAX_TARGET" \
   || ! rustup component list --toolchain "$CHARON_TOOLCHAIN" --installed 2>/dev/null \
        | grep -q '^rustc-dev'; then
    echo "    installing Rust $CHARON_TOOLCHAIN (+ $CHARON_COMPONENTS, $HAX_TARGET)"
    rustup toolchain install "$CHARON_TOOLCHAIN" --profile minimal \
        --component "$CHARON_COMPONENTS" --target "$HAX_TARGET" >/dev/null
fi
echo "    rust     $CHARON_TOOLCHAIN (for charon), target $HAX_TARGET"

HAX_BIN="${HAX_BIN:-cargo-hax}"
hax_id="$("$HAX_BIN" hax --version 2>/dev/null || true)"
if ! printf '%s\n' "$hax_id" | grep -qxE "version=$HAX_VERSION|commit=$HAX_COMMIT"; then
    if [ -z "$hax_id" ]; then
        echo "error: $HAX_BIN not found." >&2
    else
        echo "error: $HAX_BIN is not hax $HAX_VERSION ($(printf '%s\n' "$hax_id" | grep -m1 '^version=')):" >&2
    fi
    echo "       cargo install --locked cargo-hax@$HAX_VERSION --force" >&2
    exit 1
fi
echo "    hax      $HAX_VERSION ($(command -v "$HAX_BIN"))"
# Only a note: a newer release is a reason to bump (SYNC.md), not to fail.
latest="$(git ls-remote --tags https://github.com/cryspen/hax 'refs/tags/cargo-hax-v*' 2>/dev/null \
          | sed -n 's#.*refs/tags/cargo-hax-v\([0-9.]*\)$#\1#p' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
if [ -n "$latest" ] && [ "$latest" != "$HAX_VERSION" ]; then
    echo "    note: hax $latest is released; this build pins $HAX_VERSION (SYNC.md, bumping hax)"
fi

# charon and aeneas: download (checksum-verified) what hax resolves for this
# crate, then confirm it resolves the pinned versions. The Lean side must agree
# too: the generated code and the Lean libraries it imports have to come from
# the same aeneas build, so lakefile.toml and lean-toolchain are checked here.
(cd "$CRATE_ROOT" && "$HAX_BIN" hax tools install >/dev/null)
shown="$(cd "$CRATE_ROOT" && "$HAX_BIN" hax tools show)"
resolved() { printf '%s\n' "$shown" | awk -v t="$1" '$1 == t { print $2; exit }'; }
lake_rev() {   # the `rev` of the [[require]] named $1 in lakefile.toml
    awk -v n="$1" '/^\[\[require\]\]/ { hit = 0 }
                   $0 ~ "^name = \"" n "\"" { hit = 1 }
                   hit && /^rev = / { gsub(/"/, "", $3); print $3; exit }' "$PKG_DIR/lakefile.toml"
}
pin_fail=0
check_pin() {   # check_pin <what> <resolved> <expected>
    if [ "$2" != "$3" ]; then
        echo "error: $1 is '$2', expected '$3'." >&2
        pin_fail=1
    fi
}
check_pin "charon (cargo hax tools show)" "$(resolved charon)" "$EXPECT_CHARON"
check_pin "aeneas (cargo hax tools show)" "$(resolved aeneas)" "$EXPECT_AENEAS"
check_pin "lakefile.toml's aeneas rev" "$(lake_rev aeneas)" "$(resolved aeneas)"
check_pin "lakefile.toml's hax rev" "$(lake_rev hax)" "$(resolved hax-lean-lib)"
check_pin "lean-toolchain" "$(cat "$PKG_DIR/lean-toolchain")" "$(resolved lean)"
# SYNC.md records the pins for people; check it states the ones in force.
pins_table="$(sed -n '/^## Current pins/,/^## /p' "$PKG_DIR/SYNC.md")"
for v in "$HAX_VERSION" "$HAX_COMMIT" "$EXPECT_CHARON" "$EXPECT_AENEAS" "$CHARON_TOOLCHAIN" \
         "$(lake_rev hax)" "$(lake_rev CompPoly)" "$(cat "$PKG_DIR/lean-toolchain")"; do
    printf '%s\n' "$pins_table" | grep -qF "$v" || {
        echo "error: SYNC.md's \"Current pins\" table does not mention '$v'." >&2
        pin_fail=1
    }
done
if [ "$pin_fail" -ne 0 ]; then
    echo "       Look for a hax.toml above $CRATE_ROOT, or update the pins in" >&2
    echo "       extract.sh, lakefile.toml, lean-toolchain and SYNC.md together" >&2
    echo "       (SYNC.md, bumping hax)." >&2
    exit 1
fi
echo "    charon   $EXPECT_CHARON, aeneas $EXPECT_AENEAS"
echo "    lean     $(cat "$PKG_DIR/lean-toolchain") (elan installs it on first use)"
[ "$TOOLS_ONLY" = "0" ] || exit 0

# --- the patch applier, shared by both phases ---------------------------------
# `patch -F0` forbids fuzz: context drift fails loudly instead of applying
# somewhere merely plausible. Line-number offsets are still tolerated.
PATCH_FLAGS="-p1 -F0 --no-backup-if-mismatch"

# Two passes: dry-run the whole ordered set, then apply. Patches in a phase
# build on each other, so the dry run applies them to a scratch copy in order.
# $1 = phase dir, $2 = root to apply in, $3 = file recording what was applied
apply_phase() {
    local dir="$1" root="$2" record="$3"
    local pf n=0 scratch
    scratch="$(mktemp -d)"
    for pf in "$dir"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        n=$((n + 1))
    done
    if [ "$n" -eq 0 ]; then
        echo "    none"
        rm -rf "$scratch"
        return 0
    fi
    # Copy only the files the patches touch; that is all the dry run needs.
    for pf in "$dir"/[0-9]*.patch; do
        grep '^+++ ' "$pf" | sed 's/^+++ b\///; s/^+++ //; s/\t.*$//'
    done | sort -u | while read -r f; do
        mkdir -p "$scratch/$(dirname "$f")"
        [ -f "$root/$f" ] && cp "$root/$f" "$scratch/$f"
    done
    for pf in "$dir"/[0-9]*.patch; do
        if ! patch $PATCH_FLAGS -d "$scratch" < "$pf" >/dev/null 2>&1; then
            echo "error: $(basename "$pf") does not apply cleanly." >&2
            echo "       Nothing has been changed. Run patches/check-patches.sh" >&2
            echo "       to see the rejects, or fix that patch." >&2
            rm -rf "$scratch"
            return 1
        fi
    done
    rm -rf "$scratch"
    for pf in "$dir"/[0-9]*.patch; do
        patch $PATCH_FLAGS -d "$root" < "$pf" >/dev/null
        [ -z "$record" ] || echo "$pf" >> "$record"
        printf '    %-52s %s\n' "$(basename "$pf")" "[$(sed -n 's/^# Cost:[[:space:]]*//p' "$pf" | head -1)]"
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

echo "==> 1/8 checking patch conventions"
"$PKG_DIR/patches/check-patches.sh"

echo "==> 2/8 upstream tests, with and without the pre-extraction patches"
# Runs on copies of the tree (under $PRETEST_DIR), so nothing here touches the
# working tree, and the result is cached on tree + patches + script. It writes
# each patch's divergences into that patch's header.
"$PKG_DIR/patches/test-pre-patches.py"

echo "==> 3/8 pre-extraction patches (Rust source) and cargo check ($HAX_TARGET, $EXTRACT_RUSTFLAGS)"
if ls "$PRE_PATCH_DIR"/[0-9]*.patch >/dev/null 2>&1; then
    echo "    WARNING: patching Rust source changes the artifact under" >&2
    echo "             verification. See TCB.md layer 4." >&2
fi
apply_phase "$PRE_PATCH_DIR" "$REPO_ROOT" "$PRE_RECORD"
# If a patch deleted something that code reachable from p3-baby-bear still
# uses, this is where it fails. Warnings count too: a patch that leaves an
# unused import behind is not finished.
check_log="$(mktemp)"
if ! (cd "$CRATE_ROOT" && env RUSTUP_TOOLCHAIN="$CHARON_TOOLCHAIN" \
        RUSTFLAGS="$EXTRACT_RUSTFLAGS" CARGO_TARGET_DIR="$CHECK_TARGET_DIR" \
        cargo check --quiet --target "$HAX_TARGET" -p p3-baby-bear) > "$check_log" 2>&1; then
    cat "$check_log" >&2
    echo "error: the patched tree does not compile; a pre-extraction patch" >&2
    echo "       removed something that is still used." >&2
    rm -f "$check_log"
    exit 1
fi
if grep -q '^warning' "$check_log"; then
    cat "$check_log" >&2
    echo "error: the patched tree compiles with warnings." >&2
    rm -f "$check_log"
    exit 1
fi
rm -f "$check_log"
echo "    cargo check: ok, 0 warnings"

# $1 = crate dir (repo-relative), $2 = output dir, $3 = extra charon args
extract() {
    local crate="$1" out="$2" args="$3"
    # Start from nothing: generated/ holds only what this run writes.
    rm -rf "$out"
    (cd "$REPO_ROOT/$crate" && RUSTFLAGS="$EXTRACT_RUSTFLAGS" \
        "$HAX_BIN" hax into --output-dir "$out" lean \
            --charon-args="$COMMON_CHARON_ARGS $args")
    # hax scaffolds a standalone Lake package in its output dir. The package
    # here is the one this script lives in, so drop the scaffolding and the
    # ~30 MB intermediate LLBC.
    rm -rf "$out/lakefile.toml" "$out/lean-toolchain" "$out/.gitignore" \
           "$out/llbc" "$out/aeneas-error.log"
}

echo "==> 4/8 extracting with hax (backend: lean = charon + aeneas, target: $HAX_TARGET)"
# Trust the exit status: aeneas exits non-zero on any translation error, and a
# partial file is not worth building.
rm -rf "$PRISTINE"
echo "    p3-baby-bear (whole crate) -> generated/p3-baby-bear/"
extract baby-bear "$GEN_DIR/p3-baby-bear" ""
printf '%s\n' "$DEPS" | while IFS='|' read -r crate out roots; do
    echo "    $crate (scoped) -> generated/$out/"
    extract "$crate" "$GEN_DIR/$out" "$roots"
done

# Revert now, before the Lean build, so the rest of the run sees a clean tree.
# The EXIT trap stays armed and becomes a no-op.
echo "==> 5/8 restoring the Rust source"
revert_pre_patches

echo "==> 6/8 checking assumptions/ against the regenerated stubs; snapshotting"
# aeneas emits `Extraction/<X>External_Template.lean`, the declarations the
# generated code expects someone to supply, and hax seeds a fillable copy of it
# as `Assumptions/<X>External.lean` in its output dir. The filled-in copies are
# hand-written, so they live in assumptions/ (same module names; see
# lakefile.toml). Here the seeded copies are removed, after checking that each
# hand-written file still declares exactly the names the template does. The
# signatures may differ on purpose (TCB.md); `lake build` checks those.
decl_names() {
    [ -f "$1" ] || return 0
    sed -nE 's/^(noncomputable )?(axiom|opaque|def|abbrev|structure|inductive|class) ([^ :({]+).*/\3/p' "$1" | sort -u
}
stub_fail=0
for lib in $LIBS; do
    out="${lib%%|*}"; name="${lib##*|}"
    for tpl in "$GEN_DIR/$out/$name/Extraction/"*External_Template.lean; do
        [ -e "$tpl" ] || continue
        stub="$(basename "$tpl" _Template.lean)"
        mine="$ASSUME_DIR/$name/Assumptions/$stub.lean"
        rel_mine="assumptions/$name/Assumptions/$stub.lean"
        if [ ! -f "$mine" ]; then
            echo "error: the extraction needs $rel_mine, which does not exist." >&2
            echo "       Start from hax's copy: generated/$out/$name/Assumptions/$stub.lean" >&2
            stub_fail=1
            continue
        fi
        if [ "$(decl_names "$tpl")" != "$(decl_names "$mine")" ]; then
            echo "error: $rel_mine is out of date with the extraction:" >&2
            diff <(decl_names "$tpl") <(decl_names "$mine") \
                | sed -n 's/^< /         needed, not declared: /p; s/^> /         declared, no longer needed: /p' >&2
            stub_fail=1
        fi
    done
    [ "$stub_fail" -ne 0 ] || rm -rf "$GEN_DIR/$out/$name/Assumptions"
done
[ "$stub_fail" -eq 0 ] || exit 1
for lib in $LIBS; do
    out="${lib%%|*}"; name="${lib##*|}"
    mkdir -p "$PRISTINE/$out/$name/Extraction"
    cp "$GEN_DIR/$out/$name/Extraction"/*.lean "$PRISTINE/$out/$name/Extraction"/
done
echo "    ok; pristine output in generated/pristine/"

echo "==> 7/8 applying post-extraction patches (generated Lean)"
apply_phase "$POST_PATCH_DIR" "$GEN_DIR" ""

echo "==> 8/8 lake build"
# Aeneas's Lean library pulls in mathlib; without the prebuilt cache this is an
# hours-long from-source build. `cache get` is a no-op once warm.
(cd "$PKG_DIR" && lake exe cache get >/dev/null 2>&1) || \
    echo "note: 'lake exe cache get' failed; mathlib may build from source" >&2
if (cd "$PKG_DIR" && lake build); then
    n_sorry=$(cd "$GEN_DIR" && find . -path ./pristine -prune -o -name '*.lean' ! -name '*_Template.lean' -print \
                  | xargs cat | grep -c 'sorry' || true)
    n_opaque=$(cd "$ASSUME_DIR" && find . -name '*.lean' | xargs cat | grep -cE '^(noncomputable )?(axiom|opaque) ' || true)
    echo "    generated code contains $n_sorry 'sorry'; assumptions/ declares $n_opaque axiom/opaque constant(s)"
else
    echo "WARNING: lake build failed. The patched files are in place in generated/;" >&2
    echo "         inspect for drift. Fix the failing patch (its .rej shows what" >&2
    echo "         moved), or author a new one with patches/new-patch.sh." >&2
    exit 1
fi

if [ -t 1 ]; then
    light_green=$'\033[38;5;157m'; reset=$'\033[0m'
else
    light_green=; reset=
fi
echo "${light_green}Done. Build passed. ✓${reset}"
echo "${light_green}A green build means the extraction TYPE-CHECKS, meaning that"
echo "the theorems in spec/ hold up to the assumptions; See TCB.md.${reset}"
