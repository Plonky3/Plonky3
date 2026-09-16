#!/usr/bin/env bash
#
# check-patches.sh -- enforce the patch conventions.
#
# Patches are meant to be read one at a time, so every patch must carry its own
# rationale. There is deliberately no README cataloguing them: a catalogue drifts
# out of sync with the patches, whereas a header cannot. This script is what
# keeps that promise honest.
#
# Checks, per patch:
#   - filename is NNN-lowercase-slug.patch
#   - the required header fields are present and non-empty
#   - `# Patch:` matches the filename stem, `# Phase:` matches the directory
#   - `# Hunks:` matches the actual number of @@ hunks
#   - every file the diff touches is listed in `# Target:`, is relative, and
#     does not escape the phase root
# and per phase:
#   - the whole ordered set applies cleanly (--dry-run, zero fuzz)
#
# Usage: ./patches/check-patches.sh          (run from anywhere)
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$PKG_DIR/../../.." && pwd)"
PRISTINE="$HERE/post-extraction/pristine.snapshot.lean"

REQUIRED="Patch Phase Target Hunks Cost"
fail=0
err() { echo "  FAIL: $*" >&2; fail=1; }

field() {   # field <file> <name>  -> value, trimmed
    sed -n "s/^# $2:[[:space:]]*//p" "$1" | head -1 | sed 's/[[:space:]]*$//'
}

check_one() {
    local pf="$1" phase="$2" base stem
    base="$(basename "$pf")"
    stem="${base%.patch}"
    echo "  $base"

    case "$base" in
        [0-9][0-9][0-9]-*.patch) ;;
        *) err "$base: name must be NNN-slug.patch" ;;
    esac
    if ! printf '%s' "$stem" | grep -Eq '^[0-9]{3}-[a-z0-9]+(-[a-z0-9]+)*$'; then
        err "$base: slug must be lowercase alphanumeric words separated by '-'"
    fi

    local f
    for f in $REQUIRED; do
        if [ -z "$(field "$pf" "$f")" ]; then
            err "$base: missing or empty '# $f:' header"
        fi
    done

    [ "$(field "$pf" Patch)" = "$stem" ] || \
        err "$base: '# Patch:' is '$(field "$pf" Patch)', expected '$stem'"
    [ "$(field "$pf" Phase)" = "$phase" ] || \
        err "$base: '# Phase:' is '$(field "$pf" Phase)', expected '$phase'"

    # Header must not contain a line that patch would mistake for diff content.
    # The diff may start with git's own `diff --git` preamble, so treat either
    # that or the `--- ` file header as the end of the prose header.
    local first_diff
    first_diff="$(grep -nE '^(diff --git |--- )' "$pf" | head -1 | cut -d: -f1)"
    if [ -z "$first_diff" ]; then
        err "$base: no '--- ' diff header found"
        return
    fi
    if head -n $((first_diff - 1)) "$pf" | grep -qE '^(\+\+\+|@@)'; then
        err "$base: header contains a line starting with '+++' or '@@'"
    fi
    if head -n $((first_diff - 1)) "$pf" | grep -qvE '^#|^$'; then
        err "$base: header lines must start with '#' or be blank"
    fi

    local declared actual
    declared="$(field "$pf" Hunks)"
    actual="$(grep -c '^@@' "$pf" || true)"
    [ "$declared" = "$actual" ] || \
        err "$base: '# Hunks: $declared' but the diff has $actual"

    # Every touched file must be declared in Target and stay inside the root.
    local tgt p touched
    tgt="$(field "$pf" Target)"
    touched="$(mktemp)"
    grep '^+++ ' "$pf" | sed 's/^+++ b\///; s/^+++ //; s/\t.*$//' > "$touched"
    # Redirection, not a pipe: a pipe would run the loop in a subshell and its
    # `fail=1` would be discarded.
    while read -r p; do
        [ -n "$p" ] || continue
        case "$p" in
            /*|*..*) err "$base: target '$p' is absolute or escapes the root" ;;
        esac
        if ! printf '%s' "$tgt" | tr ',' '\n' \
                | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -qxF "$p"; then
            err "$base: diff touches '$p', absent from '# Target: $tgt'"
        fi
    done < "$touched"
    rm -f "$touched"
}

check_phase() {
    local phase="$1" root="$2"
    local dir
    dir="$HERE/$phase"
    echo "== $phase =="
    local n=0 pf
    for pf in "$dir"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        check_one "$pf" "$phase"
        n=$((n + 1))
    done
    if [ "$n" -eq 0 ]; then
        echo "  (none)"
        return
    fi

    # Does the ordered set apply cleanly?
    echo "  -- dry-run applying $n patch(es) in order"
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    if [ "$phase" = "post-extraction" ]; then
        if [ ! -f "$PRISTINE" ]; then
            echo "  SKIP: no pristine.snapshot.lean; run build-proofs.sh first" >&2
            return
        fi
        mkdir -p "$tmp/extraction"
        cp "$PRISTINE" "$tmp/extraction/p3_baby_bear.lean"
        root="$tmp"
    fi
    for pf in "$dir"/[0-9]*.patch; do
        [ -e "$pf" ] || continue
        if [ "$phase" = "post-extraction" ]; then
            patch -p1 -F0 --no-backup-if-mismatch -d "$root" < "$pf" >/dev/null \
                || err "$(basename "$pf") does not apply"
        else
            patch -p1 -F0 --dry-run -d "$root" < "$pf" >/dev/null \
                || err "$(basename "$pf") does not apply"
        fi
    done
}

check_phase pre-extraction  "$REPO_ROOT"
check_phase post-extraction "$PKG_DIR"

if [ "$fail" -ne 0 ]; then
    echo "check-patches: FAILED" >&2
    exit 1
fi
echo "check-patches: all patches conform"
