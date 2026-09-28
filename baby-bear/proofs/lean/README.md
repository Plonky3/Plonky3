# Hax `lean` backend (charon + aeneas)

The Lake package for the extraction of `p3-baby-bear` with hax's Lean backend,
`cargo hax into lean`, which runs [Charon](https://github.com/AeneasVerif/charon)
and [Aeneas](https://github.com/AeneasVerif/aeneas). This directory is the
package root.

## Build

```bash
./baby-bear/proofs/lean/extract.sh
```

The generated Lean is not committed, so this is the way to build from a fresh
checkout. It needs `rustup`, `cargo`, `elan` (for `lake`), `python3`, `rsync`,
`patch`, and the pinned hax on `PATH`:

```bash
cargo install --locked cargo-hax@0.4.1
```

The script checks that `cargo-hax` is that release (or uses `HAX_BIN`). hax
fetches its own charon and aeneas (checksum-verified, into `~/.cache/hax/`),
and the script installs the nightly Rust toolchain charon runs under. It checks
that `lakefile.toml`, `lean-toolchain` and the pin table in
[`SYNC.md`](SYNC.md) agree with those pins. `./extract.sh --tools-only` stops
after the tools. `--help` lists the steps.

The first run takes a while: running upstream's test suite twice (step 2; the
result is cached) and fetching mathlib. After that, `lake build` alone rebuilds
the proofs until the Rust or a patch changes. When
`baby-bear/src` (or a dependency item listed in `DEPS` in `extract.sh`)
changes, re-run `extract.sh` and follow [`SYNC.md`](SYNC.md).

## Layout

Everything in `generated/` is machine output; everything else is written by
hand.

```
lean/
  extract.sh                   the build: tools → tests → patch → extract → check → patch → lake build
  generated/                   MACHINE OUTPUT: gitignored, rewritten by every extract.sh run
    p3-baby-bear/P3BabyBear/     aeneas's translation of p3-baby-bear
      Extraction/Types.lean        the crate's structs
      Extraction/Funs.lean         everything else (consts, fns, impls)
      Extraction/*External.lean    one-line imports of the stubs in assumptions/
      Extraction/*External_Template.lean  what aeneas expects those stubs to declare; not built
    p3-monty-31/P3Monty31/       the items of p3-monty-31 that p3-baby-bear uses (scoped extraction)
    p3-mds/P3Mds/                the same for p3-mds
    pristine/                    the output before post-extraction patches
  assumptions/                 HAND-WRITTEN, TRUSTED: what the generated code needs and aeneas does not produce
    P3BabyBear/Assumptions/      stubs aeneas's output imports: the 4 opaque Debug::fmt bodies
    P3Monty31/Assumptions/       the same for p3-monty-31
    Interface/                   stand-ins for dependency code aeneas cannot translate
      P3Field.lean  P3Monty31.lean  P3Poseidon1.lean  P3Poseidon2.lean  CoreModelsExt.lean
  spec/P3BabyBearProofs/       HAND-WRITTEN: the theorems
    Constants.lean               the constants vs. CompPoly
    MontyField31.lean            MontyField31::new, the table constructors, the const assertions
  patches/                     HAND-WRITTEN diffs
    pre-extraction/              to the Rust source; applied before hax, reverted after
    post-extraction/             to generated/; applied after hax
    check-patches.sh  new-patch.sh  test-pre-patches.py
  lakefile.toml  lean-toolchain  lake-manifest.json
  README.md  TCB.md  SYNC.md
```

| Lake library | `srcDir` | Written by |
|---|---|---|
| `P3BabyBear`, `P3Monty31`, `P3Mds` | `generated/<crate>/` | aeneas, plus the post-extraction patches |
| `Assumptions` | `assumptions/` | by hand. Trusted: see `TCB.md`, layer 3 |
| `P3BabyBearProofs` | `spec/` | by hand |

The generated code imports its stubs by fixed module names
(`P3BabyBear.Assumptions.TypesExternal`, …). Those modules live in
`assumptions/` under the same names, as explicit roots of the `Assumptions`
library, so nothing hand-written sits in `generated/`. hax seeds a fresh copy
of each stub in its output on every run; `extract.sh` deletes it after checking
that the hand-written file declares exactly the names the regenerated template
does.

The import chain is: `spec/` → `P3BabyBear` → `P3BabyBear.Assumptions.TypesExternal`
→ `Interface.P3Monty31` → `P3Monty31` (the extracted dependency) and
`Interface.P3Field`/`P3Poseidon{1,2}` → `Aeneas` + `CoreModels`.

## Navigating the extraction from the Rust

Paths below are relative to `generated/p3-baby-bear/P3BabyBear/`, after a run
of `extract.sh`.

### Where a Rust item ends up

`baby-bear/src/` has four files and no public functions of substance: it is
parameter structs, trait impls that supply constants, constant tables and a
few constructors. All of it lands in **one file**, `Extraction/Funs.lean`
(~3000 lines), except the four structs, which are in `Extraction/Types.lean`.
Declarations appear in dependency order, not source order.

**Find anything by its Rust path.** Every generated declaration carries a doc
comment with its full Rust path and source span:

```lean
/-- [p3_baby_bear::baby_bear::{impl p3_monty_31::data_traits::MontyParameters for p3_baby_bear::baby_bear::BabyBearParameters}::PRIME]
    Source: 'baby-bear/src/baby_bear.rs', lines 17:4-17:34
    Visibility: public -/
```

so `grep -n "baby_bear.rs', lines 17:" generated/p3-baby-bear/P3BabyBear/Extraction/Funs.lean`
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
| `const _: () = assert!(RC.len() == …);` | `def poseidon2.const_check_N : RustM Unit := do … massert (…)` (renamed from aeneas's `__N` by post-patch 030) |
| a `while` loop (e.g. in `SAMPLING_BITS_M`) | `…SAMPLING_BITS_M_loop.body` (one iteration, returning `cont`/`done`) and `…SAMPLING_BITS_M_loop` (the `loop` over it) |

Trait *declarations* from other crates (`MontyParameters`, `MDSUtils`, …) are
Lean `structure`s whose fields are the trait's items, and supertraits are
fields named `…Inst`. Those are in `generated/p3-monty-31/P3Monty31/Extraction/Types.lean`
and `assumptions/Interface/P3Field.lean`.

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
`unfold` it by name (`spec/P3BabyBearProofs/MontyField31.lean` shows how).

### Where the dependency code lives

| Rust | Lean | How it got there |
|---|---|---|
| `p3_monty_31::data_traits::*`, `mds::MDSUtils`, `poseidon{1,2}::*Parameters` | `generated/p3-monty-31/P3Monty31/Extraction/Types.lean` | extracted (scoped) |
| `MontyField31`, `MontyField31::new`, `utils::to_monty`, default constants (`MONTY_MASK`, `BarrettParameters::N`, …) | `generated/p3-monty-31/P3Monty31/Extraction/{Types,Funs}.lean` | extracted (scoped) |
| `p3_mds::util::first_row_to_first_col` | `generated/p3-mds/P3Mds/Extraction/Funs.lean` | extracted (scoped) |
| `MontyField31::new_array`, `new_2d_array`, `no_packing` Poseidon layers | `assumptions/Interface/P3Monty31.lean` | hand-transcribed |
| p3-field's traits, `exp_1725656503` | `assumptions/Interface/P3Field.lean` | hand-written; `exp_1725656503` is `opaque` |
| `Poseidon1`, `Poseidon2`, their constructors | `assumptions/Interface/P3Poseidon{1,2}.lean` | hand-written; the constructors are `opaque` |

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
| `post-extraction/` | generated Lean, paths from **`generated/`** | after `cargo hax` | no (`generated/` is rewritten next run) |
