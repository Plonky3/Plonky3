#!/usr/bin/env bash
#
# extract.sh -- build the Lean extraction of `p3-baby-bear` from a fresh
# checkout: check the pinned tools, regenerate the Lean from the Rust with
# hax's current Lean backend (charon + aeneas), and check the proofs.
#
# The generated Lean is not committed (each `<Lib>/Extraction/` is gitignored),
# so this script is the way to build. After one run, `lake build` alone
# rebuilds the proofs.
#
#   0. tools: check (never install) the requirements listed in
#      ../README.md: `cargo-hax` is the pinned release, its charon and aeneas
#      are fetched, charon's Rust toolchain is present, and lakefile.toml,
#      lean-toolchain and SYNC.md's pin table agree
#   1. check the patch conventions (patches/check-patches.sh)
#   2. run upstream's tests with and without the pre-extraction patches, and
#      record every difference in the patch headers (patches/test-pre-patches.py)
#   3. apply patches/pre-extraction/*.patch to the RUST SOURCE, and
#      `cargo check` the variant charon extracts (the pre-patches' safety net)
#   4. run `cargo hax into lean` over the crate, and over each dependency
#      crate scoped to exactly the items p3-baby-bear uses (see DEPS below)
#   5. revert the pre-extraction patches (unconditionally, even on failure)
#   6. check the hand-written stubs in each `<Lib>/Assumptions/` against the
#      templates aeneas regenerated, then snapshot the output as .pristine/
#   7. apply patches/post-extraction/*.patch (the hand-reconciliation)
#   8. `lake build`
#
# Layout: hax's own (see README.md). This directory is `cargo hax into lean`'s
# default output directory for p3-baby-bear, and the scoped dependency runs
# write here too, one library each:
#
#   lean/                  <- Lake package root
#     P3BabyBear.lean        library root: imports Extraction and Verification
#     P3BabyBear/
#       Extraction/          <- hax, rewritten every run (gitignored)
#       Assumptions/         <- hand-written, trusted (TCB.md, layer 3)
#       Verification/        <- hand-written: Proofs, ProofObligations
#     P3Monty31/ P3Mds/      <- the same, for the scoped dependency extractions
#     .pristine/             <- the output before post-extraction patches
#     patches/               <- hand-written diffs, to the Rust and to Extraction/
#
# `patches/pre-extraction/` is NOT empty. The Lean
# backend cannot translate the generic associated types that rustc desugars
# `-> impl Iterator` trait methods into, and p3-field is full of them. The
# patches hide those items behind `cfg(hax_backend_lean)`. Two edits are
# unconditional and inert on a normal build: the cfg is declared to
# check-cfg, and some implied trait bounds are restated. Step 2 shows the
# normal build is unaffected (upstream's own tests, per test), and step 3
# has rustc confirm that nothing still reachable from p3-baby-bear depended
# on the hidden items. Each patch header gives its own argument; see
# patches/README.md and TCB.md.
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
PRE_PATCH_DIR="$PKG_DIR/patches/pre-extraction"
POST_PATCH_DIR="$PKG_DIR/patches/post-extraction"
PRISTINE="$PKG_DIR/.pristine"
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
# cannot translate is hand-written in the `Assumptions/` of the library that
# needs it. Whole-crate extraction of either crate fails (p3-field's trait
# cycle; see TCB.md). One line per crate:
#   <crate dir>|<Lean library>|<charon roots>
M31=p3_monty_31
DEPS="monty-31|P3Monty31|\
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
mds|P3Mds|--start-from p3_mds::util::first_row_to_first_col"
# Every extracted Lean library, all rooted in this directory.
LIBS="P3BabyBear P3Monty31 P3Mds"
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
# Nothing here installs anything: a missing tool is an error that names the
# requirement in baby-bear/proofs/README.md.
missing() {   # missing <what> <install command>
    echo "error: $1 is missing. Install it (see baby-bear/proofs/README.md):" >&2
    echo "       $2" >&2
    exit 1
}
for cmd in rustup cargo lake python3 rsync patch git; do
    command -v "$cmd" >/dev/null 2>&1 || missing "'$cmd'" "see the requirements table"
done

# The toolchain charon runs under, with the target it does not ship.
rustup target list --toolchain "$CHARON_TOOLCHAIN" --installed 2>/dev/null \
        | grep -qx "$HAX_TARGET" \
    && rustup component list --toolchain "$CHARON_TOOLCHAIN" --installed 2>/dev/null \
        | grep -q '^rustc-dev' \
    || missing "Rust $CHARON_TOOLCHAIN with $CHARON_COMPONENTS and $HAX_TARGET" \
        "rustup toolchain install $CHARON_TOOLCHAIN --profile minimal --component $CHARON_COMPONENTS --target $HAX_TARGET"
echo "    rust     $CHARON_TOOLCHAIN (for charon), target $HAX_TARGET"

HAX_BIN="${HAX_BIN:-cargo-hax}"
hax_id="$("$HAX_BIN" hax --version 2>/dev/null || true)"
if ! printf '%s\n' "$hax_id" | grep -qxE "version=$HAX_VERSION|commit=$HAX_COMMIT"; then
    if [ -z "$hax_id" ]; then
        echo "error: $HAX_BIN not found." >&2
    else
        echo "error: $HAX_BIN is not hax $HAX_VERSION ($(printf '%s\n' "$hax_id" | grep -m1 '^version=')):" >&2
    fi
    echo "       cargo install --locked cargo-hax@$HAX_VERSION --force  (see baby-bear/proofs/README.md)" >&2
    exit 1
fi
echo "    hax      $HAX_VERSION ($(command -v "$HAX_BIN"))"
# Only a note: a newer release is a reason to bump (SYNC.md), not to fail.
latest="$(git ls-remote --tags https://github.com/cryspen/hax 'refs/tags/cargo-hax-v*' 2>/dev/null \
          | sed -n 's#.*refs/tags/cargo-hax-v\([0-9.]*\)$#\1#p' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"
if [ -n "$latest" ] && [ "$latest" != "$HAX_VERSION" ]; then
    echo "    note: hax $latest is released; this build pins $HAX_VERSION (SYNC.md, bumping hax)"
fi

# charon and aeneas: confirm hax resolves the pinned versions and has already
# fetched them (`cargo hax tools install`, once). The Lean side must agree
# too: the generated code and the Lean libraries it imports have to come from
# the same aeneas build, so lakefile.toml and lean-toolchain are checked here.
shown="$(cd "$CRATE_ROOT" && "$HAX_BIN" hax tools show)"
fetched="$("$HAX_BIN" hax tools list 2>/dev/null)"
for tool in "charon $EXPECT_CHARON" "aeneas $EXPECT_AENEAS"; do
    printf '%s\n' "$fetched" | awk -v t="${tool%% *}:" -v v="${tool#* }" \
        '$1 == t { in_t = 1; next } /^[a-z]+:$/ { in_t = 0 }
         in_t && $1 == v && /installed/ { found = 1 } END { exit !found }' \
        || missing "hax's ${tool%% *} ${tool#* }" "(cd baby-bear && cargo hax tools install)"
done
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
# ../README.md lists the requirements, including these versions.
reqs="$(sed -n '/^## Requirements/,/^## /p' "$PKG_DIR/../README.md")"
for v in "$HAX_VERSION" "$EXPECT_CHARON" "$EXPECT_AENEAS" "$CHARON_TOOLCHAIN" \
         "$(cat "$PKG_DIR/lean-toolchain")"; do
    printf '%s\n' "$reqs" | grep -qF "$v" || {
        echo "error: the Requirements table in baby-bear/proofs/README.md does not mention '$v'." >&2
        pin_fail=1
    }
done
if [ "$pin_fail" -ne 0 ]; then
    echo "       Look for a hax.toml above $CRATE_ROOT, or update the pins in" >&2
    echo "       extract.sh, lakefile.toml, lean-toolchain, SYNC.md and ../README.md" >&2
    echo "       together" >&2
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

# $1 = crate dir (repo-relative), $2 = Lean library, $3 = extra hax args,
# $4 = extra charon args
extract() {
    local crate="$1" lib="$2" hax_args="$3" args="$4"
    # Only `Extraction/` is hax's to rewrite; start it from nothing so a
    # removed item cannot linger. Everything else under this directory hax
    # creates only when it is missing, so the hand-written files are safe.
    rm -rf "$PKG_DIR/$lib/Extraction" "$PKG_DIR/$lib/Extraction.lean"
    (cd "$REPO_ROOT/$crate" && RUSTFLAGS="$EXTRACT_RUSTFLAGS" \
        "$HAX_BIN" hax into $hax_args lean \
            --charon-args="$COMMON_CHARON_ARGS $args")
}

echo "==> 4/8 extracting with hax (backend: lean = charon + aeneas, target: $HAX_TARGET)"
# Trust the exit status: aeneas exits non-zero on any translation error, and a
# partial file is not worth building.
rm -rf "$PRISTINE"
# p3-baby-bear with hax's defaults: its default output directory is this one.
[ "$CRATE_ROOT/proofs/lean" = "$PKG_DIR" ] || {
    echo "error: extract.sh must live in <crate>/proofs/lean, hax's default output directory" >&2
    exit 1
}
echo "    p3-baby-bear (whole crate) -> P3BabyBear/Extraction/"
extract baby-bear P3BabyBear "" ""
printf '%s\n' "$DEPS" | while IFS='|' read -r crate lib roots; do
    echo "    $crate (scoped) -> $lib/Extraction/"
    extract "$crate" "$lib" "--output-dir $PKG_DIR" "$roots"
done

# Revert now, before the Lean build, so the rest of the run sees a clean tree.
# The EXIT trap stays armed and becomes a no-op.
echo "==> 5/8 restoring the Rust source"
revert_pre_patches

echo "==> 6/8 checking each Assumptions/ against the regenerated stubs; snapshotting"
# aeneas emits `Extraction/<X>External_Template.lean`, the declarations the
# generated code expects someone to supply, and hax seeds a fillable copy of it
# as `Assumptions/<X>External.lean` when that file is missing. Each filled-in
# copy must still declare exactly the names the template does. The signatures
# may differ on purpose (TCB.md); `lake build` checks those. A file hax has just
# seeded is refused too: it states its holes as `axiom`s, and this package
# states assumptions only as `opaque` constants.
decl_names() {
    [ -f "$1" ] || return 0
    sed -nE 's/^(noncomputable )?(axiom|opaque|def|abbrev|structure|inductive|class) ([^ :({]+).*/\3/p' "$1" | sort -u
}
stub_fail=0
for lib in $LIBS; do
    for tpl in "$PKG_DIR/$lib/Extraction/"*External_Template.lean; do
        [ -e "$tpl" ] || continue
        stub="$(basename "$tpl" _Template.lean)"
        mine="$PKG_DIR/$lib/Assumptions/$stub.lean"
        rel_mine="$lib/Assumptions/$stub.lean"
        if grep -qE '^axiom |^-- Seeded by hax' "$mine" 2>/dev/null; then
            echo "error: $rel_mine is hax's unfilled seed (or states an axiom)." >&2
            echo "       Fill it in, with each hole an \`opaque\` constant (TCB.md, layer 3)." >&2
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
done
[ "$stub_fail" -eq 0 ] || exit 1
for lib in $LIBS; do
    mkdir -p "$PRISTINE/$lib/Extraction"
    cp "$PKG_DIR/$lib/Extraction"/*.lean "$PRISTINE/$lib/Extraction"/
done
echo "    ok; pristine output in .pristine/"

echo "==> 7/8 applying post-extraction patches (generated Lean)"
apply_phase "$POST_PATCH_DIR" "$PKG_DIR" ""

echo "==> 8/8 lake build"
# Aeneas's Lean library pulls in mathlib; without the prebuilt cache this is an
# hours-long from-source build. `cache get` takes ~10 s even when warm, so it
# runs only while mathlib's build is missing.
if [ ! -f "$PKG_DIR/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean" ]; then
    (cd "$PKG_DIR" && lake exe cache get >/dev/null 2>&1) || \
        echo "note: 'lake exe cache get' failed; mathlib may build from source" >&2
fi
if (cd "$PKG_DIR" && lake build); then
    n_sorry=$(cd "$PKG_DIR" && find $LIBS -path '*/Extraction/*.lean' ! -name '*_Template.lean' \
                  | xargs cat | grep -c 'sorry' || true)
    n_opaque=$(cd "$PKG_DIR" && find $LIBS -path '*/Assumptions/*.lean' | xargs cat \
                  | grep -cE '^(noncomputable )?(axiom|opaque) ' || true)
    echo "    generated code contains $n_sorry 'sorry'; the Assumptions/ declare $n_opaque axiom/opaque constant(s)"
else
    echo "WARNING: lake build failed. The patched files are in place in each Extraction/;" >&2
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
echo "the theorems in P3BabyBear/Verification/ hold up to the assumptions; See TCB.md.${reset}"
