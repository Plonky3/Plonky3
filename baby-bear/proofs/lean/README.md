# Hax `lean` backend (charon + aeneas)

The Lake package for the extraction of `p3-baby-bear` with hax's Lean backend,
`cargo hax into lean`, which runs [Charon](https://github.com/AeneasVerif/charon)
and [Aeneas](https://github.com/AeneasVerif/aeneas). This directory is the
package root.

## Build

The requirements, and how to build, are in
[`../README.md`](../README.md). In short:

```bash
./baby-bear/proofs/lean/extract.sh
```

`--help` lists the steps. When `baby-bear/src` (or a dependency item listed in
`DEPS` in `extract.sh`) changes, re-run it and follow [`SYNC.md`](SYNC.md).

## Layout

This is hax's own layout, as `cargo hax into lean` creates it: this directory is
its default output directory for p3-baby-bear. Each extracted crate is one Lean
library rooted here, and under each `<Lib>/` there are three tiers:

- **`Extraction/`**: rewritten by hax on every run. Not committed (gitignored);
  `extract.sh` regenerates it.
- **`Assumptions/`**: seeded once by hax with holes for the external items its
  output refers to, then hand-written. Trusted: see `TCB.md`, layer 3.
- **`Verification/`**: hand-written specifications and proofs. hax never
  touches it.

```
lean/
  extract.sh                   the build: tools → tests → patch → extract → check → patch → lake build
  P3BabyBear.lean              library root: imports Extraction and Verification
  P3BabyBear/
    Extraction.lean            hax: imports Types and Funs
    Extraction/                hax, rewritten every run (gitignored)
      Types.lean                 the crate's four unit structs
      Funs.lean                  everything else (consts, fns, impls)
      *External.lean             one-line imports of Assumptions/*External
      *External_Template.lean    what aeneas expects Assumptions/*External to declare; not built
    Assumptions/               hand-written, trusted
      TypesExternal.lean         imports the dependency extractions and the stand-ins below
      FunsExternal.lean          the 4 opaque Debug::fmt bodies
      P3Monty31.lean             the p3-monty-31 items aeneas drops (new_array, …), built on P3Monty31.Extraction
      P3Poseidon1.lean  P3Poseidon2.lean   stand-ins for p3-poseidon1/2 (constructors opaque)
    Verification/              hand-written
    Verification.lean          imports Proofs and ProofObligations
    Verification/              hand-written
      Proofs.lean                Lean-only specs with their proofs
      ProofObligations.lean      proofs of hax_lib contracts; empty, as the Rust has none yet. Imports Proofs
  P3Monty31/                   scoped extraction of the p3-monty-31 items p3-baby-bear uses
    Extraction/                hax (gitignored)
    Assumptions/               hand-written: TypesExternal, FunsExternal, and the stand-ins they import:
      P3Field.lean               p3-field's traits
      CoreModelsExt.lean         two gaps in hax-lean's CoreModels
  P3Mds/Extraction/            scoped extraction of p3-mds (hax, gitignored)
  P3Monty31.lean  P3Mds.lean   hax's library roots for the dependencies (created by hax, gitignored)
  .pristine/                   the output before post-extraction patches (gitignored)
  llbc/                        charon's output, aeneas's input (gitignored, as hax's .gitignore has it)
  patches/                     hand-written diffs
    pre-extraction/              to the Rust source; applied before hax, reverted after
    post-extraction/             to each Extraction/; applied after hax
    check-patches.sh  new-patch.sh  test-pre-patches.py
  lakefile.toml  lean-toolchain  lake-manifest.json  .gitignore
  README.md  TCB.md  SYNC.md
```

hax creates `lakefile.toml`, `lean-toolchain`, `.gitignore`, the library roots,
`Assumptions/` and `Verification/ProofObligations.lean` only when they are
missing, so it never overwrites the hand-written versions here. `extract.sh`
checks each `Assumptions/*External.lean` against the regenerated template: it
must declare exactly the names the template does, and a file hax has just
seeded (with `axiom`s for holes) is refused until it is filled in.

Each crate's `Assumptions/` holds what *that crate's* extraction refers to but
nothing generates. That is why `P3Field` sits under `P3Monty31/` (p3-monty-31's
extraction needs it first) and `P3Monty31.lean` sits under `P3BabyBear/`
(p3-baby-bear uses those p3-monty-31 items; p3-monty-31's scoped extraction does
not produce them).

The import chain, from the bottom. An arrow means "imported by":

```
Aeneas + CoreModels
  → P3Monty31.Assumptions.{P3Field, CoreModelsExt}            hand-written
    → P3Monty31.Assumptions.{TypesExternal, FunsExternal}     stubs
      → P3Monty31.Extraction                                  generated
        → P3BabyBear.Assumptions.P3Monty31                    hand-written (+ P3Poseidon1/2)
          → P3BabyBear.Assumptions.TypesExternal              stub (+ P3Mds.Extraction)
            → P3BabyBear.Extraction                           generated
              → P3BabyBear.Verification.Proofs → ProofObligations → Verification
                → P3BabyBear                                  the library root
```

### Specifications and proofs

`Verification/` holds two kinds of theorem, which come from different places
and change for different reasons:

- **`Proofs.lean`**: properties stated directly in Lean, with no Rust
  counterpart. Each theorem is its own specification: the statement is the
  claim and the proof follows it. Theorems are named `<item>.<property>`
  after the Rust item: `BabyBearParameters::PRIME` is
  `baby_bear.BabyBearParameters.PRIME`, so its claims are
  `baby_bear.BabyBearParameters.PRIME.eq_fieldSize`, `….is_prime`, …, and
  `BabyBear::new(x)` gives `baby_bear.BabyBear.new.montgomery_form`.
- **`ProofObligations.lean`**: the answer to hax's contracts, below.

`ProofObligations.lean` is for contracts written in the Rust with
`#[hax_lib::requires]` / `#[hax_lib::ensures]`. There are none yet. If some are
added, hax extracts them to `<fn>.pre`, `<fn>.post` and `<fn>.spec` in
`P3BabyBear/Extraction/Specs.lean` and regenerates a `sorry` template of the
obligations in `P3BabyBear/Extraction/ProofObligations.lean`; their proofs go
in `Verification/ProofObligations.lean`, one `<fn>.spec.proof` each and
nothing else, so the two files can be diffed after each extraction. It
imports `Proofs.lean`, so contract proofs can reuse the hand-written
theorems; nothing in `Proofs.lean` depends on the contracts. Nothing in
`Proofs.lean` uses the three reserved names, nor `<def>.eq_<n>`, which Lean
reserves for equation lemmas.

## Navigating the extraction from the Rust

Paths below are relative to `P3BabyBear/`, after a run of `extract.sh`.

### Where a Rust item ends up

The extracted surface is four modules, `baby_bear`, `mds`, `poseidon1` and
`poseidon2`: parameter structs, trait impls that supply constants, constant
tables, and a few constructors (`default_babybear_poseidon{1,2}_*`). `lib.rs`
re-exports them; `extension_test.rs` is `cfg(test)`; the SIMD directories are
not compiled for this target. All of that lands in **one file**,
`Extraction/Funs.lean` (about 2900 lines), except the four unit structs, which
are `def … := Unit` in `Extraction/Types.lean`. Declarations appear in
dependency order, not source order.

**Find anything by its Rust path.** Every generated declaration carries a doc
comment with its full Rust path and source span:

```lean
/-- [p3_baby_bear::baby_bear::{impl p3_monty_31::data_traits::MontyParameters for p3_baby_bear::baby_bear::BabyBearParameters}::PRIME]
    Source: 'baby-bear/src/baby_bear.rs', lines 17:4-17:34
    Visibility: public -/
```

so `grep -n "baby_bear.rs', lines 17:" P3BabyBear/Extraction/Funs.lean`
jumps from a Rust line to its Lean.

### Naming

Everything is in `namespace p3_baby_bear`, and the Rust module path becomes a
dotted prefix: `p3_baby_bear::poseidon2::BABYBEAR_POSEIDON2_RC_16_INTERNAL` is
`p3_baby_bear.poseidon2.BABYBEAR_POSEIDON2_RC_16_INTERNAL`.

| Rust | Lean |
|---|---|
| `pub struct BabyBearParameters;` (unit struct) | `@[reducible] def baby_bear.BabyBearParameters := Unit` (`Types.lean`) |
| `const BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS: usize = 4;` | `def poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS : Std.Usize := 4#usize` |
| a constant computed by a call (`BabyBear::new_array([..])`) | `def … : RustM (Array (MontyField31 BabyBearParameters) 16#usize) := MontyField31.new_array …` (it can fail, so it is in the monad) |
| `impl MontyParameters for BabyBearParameters { const PRIME … }` | one `def` per item, `baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME`, then an instance `…Insts.P3_monty_31Data_traitsMontyParameters : p3_monty_31.data_traits.MontyParameters BabyBearParameters := { PRIME := ok …, … }` |
| impl of a generic trait, `impl InternalLayerBaseParameters<BabyBearParameters, 16> for …` | the generic arguments are appended to the name: `…Insts.P3_monty_31Poseidon2InternalLayerBaseParametersBabyBearParameters16` |
| `#[derive(Clone, Default, Debug, …)]` | `…Insts.CoreCloneClone`, `…Insts.CoreDefaultDefault`, `…Insts.CoreFmtDebug`, … |
| `fn exp_root_d<R: PrimeCharacteristicRing>(val: R) -> R` | `def …exp_root_d {R : Type} (p3_fieldfieldPrimeCharacteristicRingInst : p3_field.field.PrimeCharacteristicRing R) (val : R) : RustM R`: a trait bound becomes an explicit instance argument named `<crate><module><Trait>Inst` |
| `const _: () = assert!(RC.len() == …);` | the first in each module is `def poseidon{1,2}._`; the rest are `def poseidon{1,2}.const_check_N` (renamed from aeneas's `__N` by post-patch 030). `Verification/` proves the nine `const_check_N`, not the two `_` |
| a `while` loop (e.g. in `SAMPLING_BITS_M`) | `…SAMPLING_BITS_M_loop.body` (one iteration, returning `cont`/`done`) and `…SAMPLING_BITS_M_loop` (the `loop` over it) |

Trait *declarations* from other crates (`MontyParameters`, `MDSUtils`, …) are
Lean `structure`s whose fields are the trait's items, and supertraits are
fields named `…Inst`. Those are in `P3Monty31/Extraction/Types.lean`
and `P3Monty31/Assumptions/P3Field.lean`.

### Reading a body

Code is in `RustM`, a monad where `ok x` is a return value and `fail .panic` is
a panic. Every operation that can panic is a bind `let x ← …`: arithmetic
(`a + b` fails on overflow), `Array.index_usize`, and `massert c` (Rust's
`assert!`/`debug_assert!`, and the `const { assert!(..) }` blocks, which aeneas
keeps as runtime checks). Integers are `Std.U32`, `Std.Usize`, …, with
literals written `2013265921#u32`. `[T; N]` is `Array T N#usize` (length in the
type), `&[T]` is `Slice T`, `Vec<T>` is `alloc.vec.Vec T`. Shared references
are erased; `&mut` becomes a returned updated value (see
`Array.index_mut_usize`, which returns the element and a "put it back"
function).

Generated constants are `@[irreducible]`: to compute with one in a proof,
`unfold` it by name (`Verification/Proofs.lean` shows how).

### Where the dependency code lives

| Rust | Lean | How it got there |
|---|---|---|
| `p3_monty_31::data_traits::*`, `mds::MDSUtils`, `poseidon{1,2}::*Parameters` | `P3Monty31/Extraction/Types.lean` | extracted (scoped) |
| `MontyField31`, `MontyField31::new`, `utils::to_monty`, default constants (`MONTY_MASK`, `BarrettParameters::N`, …) | `P3Monty31/Extraction/{Types,Funs}.lean` | extracted (scoped) |
| `p3_mds::util::first_row_to_first_col` | `P3Mds/Extraction/Funs.lean` | extracted (scoped) |
| `MontyField31::new_array`, `new_2d_array`, `no_packing` Poseidon layers | `P3BabyBear/Assumptions/P3Monty31.lean` | hand-transcribed |
| p3-field's traits, `exp_1725656503` | `P3Monty31/Assumptions/P3Field.lean` | hand-written; `exp_1725656503` is `opaque` |
| `Poseidon1`, `Poseidon2`, their constructors | `P3BabyBear/Assumptions/P3Poseidon{1,2}.lean` | hand-written; the constructors are `opaque` |

Which dependency items are extracted is set by the `--start-from` roots in
`DEPS` in `extract.sh`. Neither dependency extracts as a whole crate yet: both
pull in p3-field's mutually recursive trait hierarchy, which aeneas rejects
(`TCB.md`, layer 5).

## `patches/`

How to author, update and bootstrap: [`patches/README.md`](patches/README.md).
Each file's header is its rationale, and `check-patches.sh` enforces that.

| Path | Applied to | When | Reverted |
|------|------------|------|----------|
| `pre-extraction/` | Rust source, paths from the **repo root** | before `cargo hax` | yes, on exit |
| `post-extraction/` | generated Lean, paths from **this package** (e.g. `P3BabyBear/Extraction/Funs.lean`) | after `cargo hax` | no (`Extraction/` is rewritten next run) |
