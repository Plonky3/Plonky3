#!/usr/bin/env bash
#
# new-patch.sh -- capture the current deviation as a new, self-documenting patch.
#
#   ./patches/new-patch.sh post-extraction 070-my-fix
#   ./patches/new-patch.sh pre-extraction  010-work-around-hax-ice
#
# Patches here are hand-owned, not regenerated: this script writes a patch once,
# with a header skeleton to fill in, and never touches existing ones. That is
# what lets each patch carry its own rationale (see check-patches.sh).
#
# post-extraction: reconstructs the baseline (pristine hax output + the patches
#   that already exist) in a temp dir and diffs the live file against it, so the
#   captured delta is ONLY your new change.
# pre-extraction: wraps `git diff` over the Rust tree at the repo root.
#
# The new number must sort after every existing patch in the phase, because the
# baseline is "all existing patches applied".
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$PKG_DIR/../../.." && pwd)"
PRISTINE="$HERE/post-extraction/pristine.snapshot.lean"
LIVE="$PKG_DIR/extraction/p3_baby_bear.lean"

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <pre-extraction|post-extraction> <NNN-slug>" >&2
    exit 2
fi
PHASE="$1"
STEM="$2"
case "$PHASE" in
    pre-extraction|post-extraction) ;;
    *) echo "error: phase must be pre-extraction or post-extraction" >&2; exit 2 ;;
esac
if ! printf '%s' "$STEM" | grep -Eq '^[0-9]{3}-[a-z0-9]+(-[a-z0-9]+)*$'; then
    echo "error: '$STEM' must look like 070-lowercase-slug" >&2
    exit 2
fi

DIR="$HERE/$PHASE"
OUT="$DIR/$STEM.patch"
[ ! -e "$OUT" ] || { echo "error: $OUT already exists" >&2; exit 1; }

# The new patch must sort last.
NUM="${STEM%%-*}"
for existing in "$DIR"/[0-9]*.patch; do
    [ -e "$existing" ] || continue
    e="$(basename "$existing")"
    if [ "${e%%-*}" \> "$NUM" ] || [ "${e%%-*}" = "$NUM" ]; then
        echo "error: $e sorts at or after $NUM; pick a higher number." >&2
        echo "       The baseline assumes every existing patch is already applied." >&2
        exit 1
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
RAW="$TMP/raw.diff"

if [ "$PHASE" = "post-extraction" ]; then
    [ -f "$PRISTINE" ] || { echo "error: no $PRISTINE; run build-proofs.sh first" >&2; exit 1; }
    [ -f "$LIVE" ]     || { echo "error: no $LIVE" >&2; exit 1; }
    mkdir -p "$TMP/base/extraction"
    cp "$PRISTINE" "$TMP/base/extraction/p3_baby_bear.lean"
    for pf in "$DIR"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        patch -p1 -F0 --no-backup-if-mismatch -d "$TMP/base" < "$pf" >/dev/null \
            || { echo "error: existing $(basename "$pf") no longer applies to pristine." >&2
                 echo "       Fix that patch before authoring a new one." >&2; exit 1; }
    done
    diff -u -L a/extraction/p3_baby_bear.lean -L b/extraction/p3_baby_bear.lean \
        "$TMP/base/extraction/p3_baby_bear.lean" "$LIVE" > "$RAW" || true
else
    ( cd "$REPO_ROOT" && git diff -- . ) > "$RAW" || true
fi

if [ ! -s "$RAW" ]; then
    echo "error: nothing to capture -- no deviation found." >&2
    [ "$PHASE" = "post-extraction" ] \
        && echo "       Edit extraction/p3_baby_bear.lean first." >&2 \
        || echo "       Edit the Rust source first." >&2
    exit 1
fi

HUNKS="$(grep -c '^@@' "$RAW" || true)"
TARGETS="$(grep '^+++ ' "$RAW" | sed 's/^+++ b\///; s/^+++ //; s/\t.*$//' \
           | paste -sd, - | sed 's/,/, /g')"

{
    printf '# Patch:     %s\n' "$STEM"
    printf '# Phase:     %s\n' "$PHASE"
    printf '# Target:    %s\n' "$TARGETS"
    printf '# Hunks:     %s\n' "$HUNKS"
    printf '# Cost:      TODO -- write "none", or name the axiom/assumption added\n'
    printf '# Upstream:  TODO -- delete this line, or name the upstream bug\n'
    printf '# Drop when: TODO -- the condition under which this patch can be deleted\n'
    printf '#\n'
    printf '# TODO: why this change is necessary, and why it is the least-bad option.\n'
    printf '# A reviewer should be able to judge this patch from its header alone.\n'
    printf '#\n'
    cat "$RAW"
} > "$OUT"

echo "==> wrote $OUT ($HUNKS hunk(s), targets: $TARGETS)"
echo "    Now fill in the TODO header fields, then run:"
echo "      ./patches/check-patches.sh"
