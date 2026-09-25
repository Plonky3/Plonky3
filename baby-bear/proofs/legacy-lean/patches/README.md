# Patches

There are two kinds of patches.

| | `pre-extraction/` | `post-extraction/` |
|---|---|---|
| Patches | Rust source (`baby-bear/src`, or any dependency crate) | hax's generated Lean (`extraction/*.lean`) |
| Applied | **before** `cargo hax` runs | **after** `cargo hax` runs |
| Reverted | yes — unconditionally, on exit | no, the edit is the artifact |
| Checked in | the `.patch` files | the `.patch` files (not `pristine.snapshot.lean`) |
| Trust cost | **changes the artifact under verification** | changes only the Lean encoding of it |
| Currently | **empty, deliberately** | 6 patches, `010-`..`060-` (7 hunks total) |

A **pre-extraction** patch says *hax (and/or Aeneas) could not cope with this Rust, so we
had to modify the source Rust*. That is strictly worse: the thing verified is no longer the
thing that ships, and no amount of Lean-side rigour recovers the gap. Anything
proved downstream holds of the patched crate, not of `p3-baby-bear` as published.
Pre-extraction patches should be highly scrutinized, ideally with differential testing.

A **post-extraction** patch says *hax translated this badly and we fixed the
translation*. These may be due to incomplete extraction, or a complete extraction
that does not typecheck. This also includes adding axiomatized interfaces that describe
the boundary of verification. The Rust that ships is still exactly the Rust that was
extracted, so the patch is a claim about the encoding. A reviewer checks it by reading the
diff against `pristine.snapshot.lean`. Post-extraction patches should be highly scrutinized.

## Use

```bash
../build-proofs.sh     # apply both phases (then lake build)
./check-patches.sh     # conventions + dry-run; also step 0 of the above
```

Patches apply in filename order (`010-`, `020-`, …), all-or-nothing, with
`patch -p1 -F0` (no fuzz). Pre-extraction paths are from the **repo root**;
post-extraction paths are from **this package**. Do not apply by hand unless
debugging a reject.

After any run, `git status --porcelain baby-bear/src` must be empty. The revert
is `patch -R` of exactly what was applied — never `git checkout`.

## Update

Patches are not regenerated. `./new-patch.sh` never overwrites.

| Situation | Action |
|-----------|--------|
| New deviation | Edit the live file, then `./new-patch.sh post-extraction 070-short-slug` (number must sort last). Fill the TODO header. |
| A patch no longer applies | Edit **that** `.patch` until `./check-patches.sh` is green. `.rej` files show what moved. |
| A patch is obsolete | Delete it. Shrinking the set is the goal. |

Preserve intent, not hunks. If an obligation now closes by `rfl`, drop the patch.

In generated Lean: mark every edit `-- PATCHED`; keep a replaced tactic as
`-- (by rfl)` on the next line. Prefer fixing `extraction/p3_baby_bear/*.lean`
over adding a patch. Never `sorry`.

Header fields: `Patch`, `Phase`, `Target`, `Hunks`, `Cost` (required);
`Upstream`, `Drop when` when they apply. One concern per file;
`NNN-lowercase-slug.patch`, no spaces.

## Bootstrap

No patches yet:

1. `../build-proofs.sh` — apply is a no-op; writes `pristine.snapshot.lean`.
2. Hand-edit `extraction/p3_baby_bear.lean` until `lake build` is green.
3. `./new-patch.sh post-extraction 010-first-fix`
4. Fill the header; `./check-patches.sh`.
