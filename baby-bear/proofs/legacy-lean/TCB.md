> **This document is AI-generated and agent-maintained.**

# The Trusted Codebase

Everything that, if wrong, could compromise this extraction *without* a Lean
error to flag it.

> Almost every declaration in this directory is on the **trusted** side. The
> exception is `spec/p3_baby_bear_proofs/`: five theorems about the field
> constants, which are proved rather than assumed — see "The five theorems are
> NOT in the TCB" below. Everything else is counted here.

Measured totals (regenerate with the commands in [`SYNC.md`](SYNC.md)):

| Quantity | Value |
|---|---|
| Generated `p3_baby_bear.lean` | **4306** lines, **0** `sorry` |
| Hand-written interface `extraction/p3_baby_bear/*.lean` | **1025** lines, **0** `sorry`, **17** `opaque` |
| Pre-extraction patches (Rust source) | **0** — none needed |
| Post-extraction patches | **6** files, **7** hunks total |
| `of_isOk` obligations discharged by `rfl` | **45 of 46** |
| `of_isOk` obligations needing `native_decide` | 1 of 46 |
| **`lake build` warnings** | **0** |

**There is no `sorry` anywhere in this directory**, and a clean `lake build`
emits no warnings. Every assumption is a named `opaque` constant, listed in
layer 3.

That is a real change in logical content, not tidying. `sorry` elaborates to
`sorryAx`, which can inhabit `False` — `theorem bad : False := sorry` compiles.
An interface built on `sorryAx` is formally inconsistent even when nothing
exploits it. An `opaque` constant asserts only that *some* inhabitant of its
type exists, which for a function type is true and is exactly what
"axiomatized interface" should mean. Measured:

| Form | `#print axioms` | Can inhabit `False`? | Code generator |
|---|---|---|---|
| `sorry` | `sorryAx` | **yes** | fine |
| `axiom f : T` | `[f]` | no, for data | **breaks** ("not supported by code generator") |
| `opaque f : T` | *nothing* | no | fine |

`opaque` is used rather than `axiom` because `axiom` breaks the code generator,
which this build needs for the `native_decide` obligation in patch `040`.

The practical benefit is as large as the logical one: with the previous `sorry`
warnings gone, a `sorry` introduced later by drift is immediately visible
instead of lost in noise.

What this does **not** mean: the operations are still undefined. It means the
assumption is well-formed rather than contradictory. The footprint is now
honest and small:

```
#print axioms …Impl_3.MONTY_GEN_hoisted
  -- depends on axioms: [propext]                      (was: [propext, sorryAx])
#print axioms …Impl_9.SAMPLING_BITS_M_hoisted
  -- depends on axioms: [propext, Classical.choice, Quot.sound,
  --                     …native_decide.ax_13]
```

## Layer 1 — the Lean toolchain

Pin: `lean-toolchain` = `leanprover/lean4:v4.30.0`. This is one minor version
ahead of the Hax legacy-lean proof library's own pin (`v4.29.0-rc1`); the bump is
required by CompPoly/mathlib (Layer 6) and is the same combination
`Plonky3Verif/koala-bear/proofs-legacy` already builds. The legacy-lean library
compiles cleanly under it apart from two upstream lint warnings in
`Hax/rust_primitives/USize64.lean` (a `@[reducible]` lint new in Lean 4.30),
which are not ours to fix. The specific tag is not itself part of the TCB — we
trust Lean as a system, not a release — but it must be an official
`leanprover/lean4` tag. Build tool: `lake`.

## Layer 2 — the Hax proof library

Source `https://github.com/cryspen/hax`, subdir `hax-lib/proof-libs/legacy-lean`,
pinned in `lake-manifest.json` at `e116155418076d3ede11d7f4993e6602cdcaf2cd`
(`Qq` at `23324752757bf28124a518ec284044c8db79fee5`). It supplies the Lean
shadows of Rust: `RustM`, `RustArray`, `RustSlice`, the integer types, and the
`core_models.*` models of `core`/`alloc`. Soundness rests on those modelling
Rust faithfully.

The lakefile pins Hax by **revision, not `main`**: the generated file and the
proof library must come from the same hax revision, and a floating `rev` would
let them drift apart silently.

## Layer 3 — the hand-written interface (`extraction/p3_baby_bear/`) (1025 lines, 17 `opaque`, 0 `sorry`)

One file per crate in `baby-bear/Cargo.toml`'s `[dependencies]`, so the
axiomatization boundary is exactly the crate's declared dependency edge.

### Given real, faithful bodies

Transcriptions of upstream, not placeholders. Supplying these is what lets 45 of
46 `of_isOk` obligations discharge by `rfl` — the whole `(by rfl)` → `(by sorry)`
patch category disappeared.

**These transcriptions are now unchecked.** `SanityCheck.lean`, which verified
them against upstream's own `const assert!`s and against the mathematical specs,
has been removed. Nothing in the pipeline would notice if one of these drifted
from upstream; only `PRIME`, `MONTY_BITS` and `TWO_ADICITY` are covered, by the
theorems in `spec/`. Re-reading them against upstream is a manual audit step.

| Symbol | Upstream | Checked by |
|---|---|---|
| `utils.to_monty` | `utils.rs:7-9` | **nothing** — read it against upstream by hand |
| `Impl.new` | `monty_31.rs:51` | **nothing** |
| `Impl.new_array`, `Impl.new_2d_array` | `monty_31.rs:88`, `101` | the length assertions in the extraction |
| `MontyParameters.MONTY_MASK` | `data_traits.rs:23` | **nothing** |
| `TwoAdicData.ODD_FACTOR` | `data_traits.rs:93` | **nothing** |
| `BarrettParameters.PRIME_I128`, `.PSEUDO_INV` | `data_traits.rs:54-55` | **nothing** |
| `p3_field.dup.Dup.dup` | blanket `impl<T: Copy>`, body `*self` | faithful by inspection |
| `hax_ext` `AsRef.as_ref` | `rust_primitives.unsize` **is** this coercion | faithful by inspection |

`new_array`/`new_2d_array` use `Vector.ofFn`, not `Vector.map`: `Array.map`'s
`size` does not reduce definitionally, which blocks the length assertions. This
is load-bearing.

`MONTY_MASK`, `ODD_FACTOR`, `PRIME_I128` and `PSEUDO_INV` are written with total
arithmetic rather than hax's fallible `<<<?` / `cast_op`. These are class
*defaults*, generic in `Self`, so a fallible operation's `of_isOk` obligation
cannot be discharged for an abstract `MONTY_BITS`. Total arithmetic also avoids
needing `Cast u32 i128` / `Cast i128 i64` instances, which the Hax library does
not provide.

### The 10 assumed operations

All `opaque`. Nothing else in this directory is assumed.

| Count | Symbol(s) | What is assumed | Reachable? |
|---:|---|---|---|
| 6 | `p3_field.field.ring_{add,sub,mul,add_assign,sub_assign,neg}` | **The core assumption: no ring arithmetic is modelled.** `Output := R` is faithful; the operations are undefined. | yes |
| 1 | `p3_field.exponentiation.exp_1725656503` | BabyBear's `exp_root_d` / seventh-root map. Upstream is the addition chain `x^1725656503`; nothing about the S-box follows from the opaque. | yes |
| 1 | `p3_mds.util.first_row_to_first_col` | Circulant first-row → first-column (`col[0] = row[0]`, `col[i] = row[N-i]`). A commented hand-model is in `p3_mds.lean`; the live symbol is opaque, so the MDS coefficients are not pinned. | yes |
| 1 | `p3_monty_31.data_traits.mul_w_default` | Upstream body is `a * Self::W`, needing `Algebra`'s multiplication — one of the six above. Used at `DEG` 4 and 8. | yes |
| 1 | `p3_monty_31.mds.mds_permute_mut` | The MDS permutation's behaviour. | yes |

Seven further `opaque`s are structural rather than behavioural — four instance
witnesses (`mds.Impl`, `mds.Impl_Permutation` and their two `AssociatedTypes`
companions) and three Poseidon1/2 constructors (`p3_poseidon1.Impl_1.new`,
`p3_poseidon2.Impl.new`, `p3_poseidon2.Impl_4.new`). They assert an inhabitant
exists without saying which. 10 behavioural + 7 structural = **17** `opaque`
declarations; recount with the regex in [`SYNC.md`](SYNC.md) step 3, since a
bare `grep -c opaque` also matches the prose in these files.

The `Permutation` witnesses supply their `Clone`/`Sync` parent clauses
*explicitly* from the Hax library's blanket instances. Hax's `Copy` carries
`Clone` as a parent clause, so leaving them to instance search sends the
elaborator around a `Clone`/`Copy` cycle until it times out.

## Layer 4 — the patches

Patches come in two kinds, and they are kept in separate directories because
their trust costs are not comparable:

- **`patches/pre-extraction/` — EMPTY, and that is an assertion.** A patch here
  modifies the Rust source before hax runs, which means the artifact verified is
  no longer the artifact that ships. Nothing proved on the Lean side can repair
  that gap. There are none, because the SIMD backends are excluded by choosing a
  SIMD-free extraction target rather than by patching out `cfg`s (see "The
  extraction is target-scoped" below). **If this directory ever becomes
  non-empty, every entry needs a justification recorded here, and every
  downstream claim weakens from "about `p3-baby-bear`" to "about a patched
  `p3-baby-bear`."**
- **`patches/post-extraction/` — six patches, 7 hunks total.** These change
  only hax's generated Lean, so the Rust under extraction is untouched and the
  patch is a claim about the *encoding*. That is what the rest of this section
  accounts for.

### The six post-extraction patches

One logical change per file, each carrying its own rationale in its own header.
There is deliberately **no README cataloguing them**: a catalogue drifts out of
sync with the patches, a header cannot. `patches/check-patches.sh` enforces that
every patch declares `Patch`, `Phase`, `Target`, `Hunks` and `Cost`, that the
declared hunk count matches the diff, and that the ordered set applies with
**zero fuzz**. `build-proofs.sh` runs it before applying anything, and echoes
each patch's `Cost:` as it applies, so the one costly patch is visible on every
build.

| Patch | Hunks | Cost |
|---|---:|---|
| `010-import-axiomatized-interface` | 1 | none |
| `020-raise-maxrecdepth` | 1 | none |
| `030-babybear-abbrev-instance-binders` | 1 | none |
| `040-sampling-bits-native-decide` | 1 | **`Lean.ofReduceBool`** |
| `050-name-anonymous-const-assertions` | 2 | none |
| `060-qualify-internal-layer-mat-mul` | 1 | none |

Six of the seven hunks are name and elaboration plumbing with no soundness cost;
030, 050 and 060 work around hax bugs. **`040` is the only judgement call**, and
it is isolated in its own file precisely so it can be read alone:

`SAMPLING_BITS_M` is built by a Rust `const while` loop, which hax compiles to
`Loop.MonoLoopCombinator.while_loop` — a monotone fixpoint the kernel cannot
reduce, so `rfl` fails. The loop is bounded and total, and it *does* evaluate,
so `native_decide` discharges it. The cost is the `Lean.ofReduceBool` axiom:
the answer is computed by the compiled evaluator rather than checked by the
kernel. That is a different trust assumption from a `sorry`, not obviously a
smaller one — it trusts Lean's compiler and runtime, but unlike a `sorry` it
does establish the fact. Reverting to `(by sorry)` is a one-line change to that
hunk if you would rather keep the kernel as the only checker.

Note also that hax emits a **constant-zero termination measure**
(`from_machine (0 : u32)`) for this loop, which makes the library's own
`while_loop.spec` unusable here — its `step` hypothesis would require `0 < 0`.
A kernel-checked proof is therefore possible but needs a bespoke argument
against `Loop.MonoLoopCombinator` with a real measure (`64 - k`); that is the
clean way to remove this hunk.

## Layer 5 — the hax extractor

`cargo hax into legacy-lean`. Two assumptions: that hax's translation of Rust to
Lean is faithful, and that its item selection did not silently drop anything
that matters. hax labels this backend **legacy and experimental**. Of the five
trust layers this is the one deserving the most scrutiny, and the least
mechanically checkable.

Not load-bearing for soundness: the `-Zcrate-attr=feature(maybe_uninit_slice)`
flag, and the pinned nightly.

### Extractor bugs found while building this

Both are worked around in the patch and are worth reporting upstream:

1. **Anonymous consts.** Rust's `const _: () = assert!(..)` (baby-bear's
   poseidon table length checks) is emitted as `def _`, and `_` is not a valid
   Lean identifier — a hard parse error. 2 sites.
2. **Hoisted-helper name collision.** hax emits
   `def Impl_2.internal_layer_mat_mul_hoisted` and then references
   `Impl_2.internal_layer_mat_mul_hoisted` from *inside* the instance also named
   `Impl_2`, so Lean resolves it as a self-projection and fails. 1 site.

## The extraction is target-scoped

Extracted for `thumbv7em-none-eabi`, so all three SIMD backends and monty-31's
packing modules are `cfg`-excluded and the semantics are monty-31's portable
`no_packing` path. This buys a patch-free Rust tree, host-independent output,
and removal of the SIMD intrinsic axioms — but it is an assumption: **nothing
here says anything about the AVX2, AVX-512 or NEON code paths**, which is what
production actually runs on x86-64 and aarch64.

## Layer 6 — the specification: CompPoly and mathlib

`lakefile.toml` requires CompPoly at `rev = "v4.30.0"`
(`18c1613e9186c7afa79a4179eea5f4b80d8e9e00`), which pulls mathlib `v4.30.0`
(`c5ea00351c28…`) and eight inherited packages. All are pinned in
`lake-manifest.json`.

**CompPoly and mathlib are trusted.** `spec/p3_baby_bear_proofs/constants.lean`
proves the extracted constants equal *CompPoly's* definitions of them, so
CompPoly is the yardstick, and a wrong yardstick yields a wrong result. What is
actually relied on is small and inspectable — from `CompPoly/Fields/BabyBear.lean`:

| Used | Definition |
|---|---|
| `BabyBear.fieldSize` | `2 ^ 31 - 2 ^ 27 + 1` (`@[reducible]`) |
| `BabyBear.twoAdicity` | `27` (`@[reducible]`) |

Both are literal definitions, not derived results, so the trust here is the
trivial kind: that someone wrote the right numbers down in CompPoly. Mathlib
enters only through `Nat.Coprime`. `BabyBear.is_prime` (a Pratt certificate) is
*not* currently used — the constants spec does not yet assert primality.

## The five theorems are NOT in the TCB

This is the distinction that matters. Everything above is assumed. The contents
of `spec/p3_baby_bear_proofs/` are the opposite: they are the **verification target**,
the first statements in this tree that are discharged rather than admitted.

```
monty_prime_eq_fieldSize        depends on axioms: [propext]
two_adicity_eq_spec             depends on axioms: [propext, Classical.choice, Quot.sound]
monty_bits_eq_thirtyTwo         depends on axioms: [propext]
coprime_seven_pred_prime        depends on axioms: [propext]
fieldSize_sub_one_factorization' depends on axioms: [propext, Classical.choice, Quot.sound]
```

No `sorryAx`, and nothing from the Layer 3 interface — the `opaque` declarations
contribute no axioms, which is exactly why they were written that way.

Two cautions on reading them:

- `coprime_seven_pred_prime` uses **7**, not the **3** that KoalaBear uses.
  `p - 1 = 2^27 * 3 * 5`, so `gcd(3, p - 1) = 3`: the cube map is *not* a
  bijection on BabyBear's unit group, and the KoalaBear theorem transplanted
  unchanged would be false. `decide` refutes it outright. The extraction declares
  `RelativelyPrimePower BabyBearParameters ((7 : u64))`, which is what is checked.
- `fieldSize_sub_one_factorization'` relates two *extracted* constants to each
  other, with the odd part `15` supplied by us. Its independent content comes
  from `two_adicity_eq_spec` tying `TWO_ADICITY` to CompPoly's `twoAdicity`.
  Read the pair together, not the factorization alone.

## What is *not* in the TCB

- The Rust source of `p3-baby-bear` — that is the object under extraction.
- The numerical constants — they are extracted data, checkable against upstream.
- Anything in `.lake/` — build output.
- The five theorems in `spec/p3_baby_bear_proofs/` — they are proved, not assumed.
- `patches/post-extraction/pristine.snapshot.lean` — regenerated hax output, kept
  only so the patch can be diffed and reviewed.

## How the TCB shrinks

The three live directions, in order of value:

1. **Model the six ring operations.** They are the core assumption and they also
   unblock `mul_w_default`. Everything about BabyBear arithmetic is downstream.
2. **Extract `p3-monty-31` for real** and consume it as a Lake dependency (the
   pattern `keccak`/`blake3` use for `p3_symmetric` in the fork). That collapses
   the 666-line monty-31 file and makes `first_row_to_first_col` and
   `exp_1725656503` real instead of opaques. Uncommenting the hand-model in
   `p3_mds.lean` is the smaller local step for the MDS coefficients.
3. **Upstream the two hax bugs** (`def _`, the hoisted-helper name collision),
   each worth one patch hunk.

Banked so far, worth keeping as the regression baseline:

| | before | after |
|---|---|---|
| `sorry` in the generated file | 40 | **0** |
| `sorry` in the interface | 19 | **0** |
| `lake build` warnings | 17 | **0** |
| post-extraction patch | 445 lines / 42 hunks | **6 files / 7 hunks** |
| `sorryAx` in the axiom footprint | yes | **no** |

## Audit checklist

- [ ] `lean-toolchain` is an official `leanprover/lean4` tag.
- [ ] The Hax `rev` in `lake-manifest.json` is a fixed SHA, not `main`, and
      matches the hax that produced `p3_baby_bear.lean`.
- [ ] `lake build` from a clean `.lake/build` emits **0 warnings, 0 errors**.
- [ ] `grep -c sorry` is **0** in both `extraction/p3_baby_bear.lean` and
      `extraction/p3_baby_bear/`
      (comments in `p3_field.lean` mention the word; no declaration uses it).
- [ ] Every `opaque` in `p3_baby_bear/` appears in a layer-3 table.
- [ ] `patches/check-patches.sh` exits 0 — every patch declares `Cost:`, its
      declared hunk count matches its diff, and the set applies with zero fuzz.
- [ ] `040-sampling-bits-native-decide` is the ONLY patch whose `Cost:` is not
      `none`. Grep it: `grep -h '^# Cost:' patches/*/[0-9]*.patch`.
- [ ] `build-proofs.sh` ends green through step 6/6.
- [ ] `git status --porcelain baby-bear/src monty-31/src` is empty afterwards.
- [ ] The figures at the top of this file match a fresh `wc -l` / `grep -c`.
