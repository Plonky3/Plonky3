> **This document is AI-generated and agent-maintained.**

# The Trusted Codebase

Everything that, if wrong, could compromise this extraction *without* a Lean
error to flag it. Machine output lives in `generated/` (not committed;
`extract.sh` rewrites it); what is written by hand is `assumptions/`, `spec/`
and `patches/`.

Measured totals (regenerate with the commands in [`SYNC.md`](SYNC.md)):

| Quantity | Value |
|---|---|
| Generated for p3-baby-bear (`generated/p3-baby-bear/…/{Types,Funs}.lean`) | **2947** lines, **0** `sorry` |
| Generated for dependencies (`generated/p3-{monty-31,mds}/…/{Types,Funs}.lean`) | **568** lines, **0** `sorry` |
| Hand-written assumptions (`assumptions/`) | **522** lines, **0** `sorry`, **0** `axiom`, **10** `opaque` |
| Pre-extraction patches (Rust source) | **3** files, **17** hunks. Hidden items are behind `cfg(hax_backend_lean)`; the cfg declaration and some redundant bounds are not |
| Upstream tests diverging under the pre-patches | **0** (4671 run) |
| Post-extraction patches | **3** files, **17** hunks |
| Theorems in `spec/` | **27**, axiom footprint `[propext, Classical.choice, Quot.sound]` |
| **`lake build` warnings** | **0** |

There is no `sorry`, no `axiom` and no `native_decide` in anything the build
compiles. (aeneas regenerates `Extraction/FunsExternal_Template.lean` with four
`axiom`s on every run; nothing imports it, so it is never built. The imported
module, `assumptions/stubs/P3BabyBear/Assumptions/FunsExternal.lean`, states them as
`opaque`.) Every assumption is a named `opaque` constant (layer 3). No theorem
in `spec/` depends on any of them: `#print axioms` on each shows only Lean's
three standard axioms.

## Layer 1 — the Lean toolchain

`lean-toolchain` = `leanprover/lean4:v4.31.0`, the version hax 0.4.1 resolves
(`cargo hax tools show`). It is an official `leanprover/lean4` tag. Build tool:
`lake`.

## Layer 2 — the Lean libraries the generated code imports

| Library | Pin | Supplies |
|---|---|---|
| `aeneas` (`cryspen/aeneas`, `backends/lean`) | `build-183e4f0` | `Aeneas.Std`: `RustM`, scalars, `Array`, `Slice`, the `⦃ ⦄` spec logic and `step` |
| `hax` (`cryspen/hax-lean`) | `v0.3.27` | `CoreModels`: the Lean models of `core`/`alloc` that aeneas's output names |

Both are pinned to exactly what hax 0.4.1 resolves. The generated files and
the libraries must come from the same aeneas build, and a floating `rev` would
let them drift apart silently. Soundness rests on these modelling Rust
faithfully. `assumptions/Interface/CoreModelsExt.lean` fills two gaps in
`CoreModels` (below).

## Layer 3 — the dependency interface

aeneas emits only the crate it runs on. Everything p3-baby-bear's output names
in `p3-monty-31`, `p3-mds`, `p3-field`, `p3-poseidon1` and `p3-poseidon2`
comes from one of two places.

### Extracted: `generated/p3-monty-31/`, `generated/p3-mds/` (trusted like the main extraction, not hand-written)

Scoped runs of the same pipeline over the dependency crate, rooted with
`--start-from` at exactly the items p3-baby-bear refers to (`DEPS` in
`extract.sh`). They are regenerated on every build, so drift in a
dependency's declarations or bodies is caught, not assumed away.

| Crate | Roots | What comes out |
|---|---|---|
| `p3-monty-31` | the `data_traits`/`mds`/`poseidon{1,2}` parameter traits, `MontyField31::new`, its `Clone` impl | every trait declaration p3-baby-bear implements, their default constants, `MontyField31`, **`new` and `utils::to_monty` with real bodies** |
| `p3-mds` | `util::first_row_to_first_col` | the function with its real loop |

### Hand-written: `assumptions/`

| File | What | Why not extracted |
|---|---|---|
| `Interface/P3Field.lean` | `PrimeCharacteristicRing` (the 7 members used), `Field`, `PrimeField` (memberless), `UniformSamplingField`, `dup::Dup`; `exp_1725656503` (**opaque**) | p3-field's trait hierarchy is mutually recursive through its associated types, and a scoped run pulls in default-method bodies that fail |
| `Interface/P3Poseidon1.lean` | `Poseidon1`, `Poseidon1Constants` (complete), the two layer-constructor traits; `FullRoundConstants`, `PartialRoundConstants` (**opaque** types); `Poseidon1::new` (**opaque**) | aeneas internal error in `core::iter` |
| `Interface/P3Poseidon2.lean` | `Poseidon2`, `ExternalLayerConstants`, the two layer-constructor traits (copied from aeneas's partial translation); `Poseidon2::new`, `ExternalLayerConstants::new` (**opaque**) | aeneas internal error on `Poseidon2::new` |
| `Interface/P3Monty31Missing.lean` | `new_array`, `new_2d_array` (transcribed: elementwise `new`); the `no_packing` Poseidon layer types and their four constructor impls (transcribed: one struct literal each); memberless `Field`/`PrimeField` witnesses | aeneas drops `new_array`/`new_2d_array` without a diagnostic; the rest was not rooted |
| `Interface/CoreModelsExt.lean` | `Clone for PhantomData<T>` (identity), `AsRef<U> for &T` (delegation) | missing from hax-lean `CoreModels` |
| `stubs/P3BabyBear/Assumptions/FunsExternal.lean` | the four `Debug::fmt` bodies (**opaque**) | charon is told to make them opaque (below) |
| `stubs/P3BabyBear/Assumptions/TypesExternal.lean`, `stubs/P3Monty31/Assumptions/*.lean` | imports only | the generated code imports these module names; they point it at `Interface/` |

The `…/Assumptions/` files carry the module names aeneas's output imports,
and `extract.sh` fails if a regenerated `Extraction/*External_Template.lean`
declares different names than its hand-written counterpart.

The trait structures declare only the members the generated code projects.
Instances are passed in as parameters, so anything proved about code generic
over, say, `PrimeCharacteristicRing` holds for every implementation. The
opaque items are the whole of what is *assumed*:

| Opaque | Reached by `spec/`? |
|---|---|
| `p3_field.exponentiation.exp_1725656503` | no |
| `p3_poseidon1.Poseidon1.new`, `…FullRoundConstants`, `…PartialRoundConstants` | no |
| `p3_poseidon2.Poseidon2.new`, `…ExternalLayerConstants.new` | no |
| four `Debug::fmt` bodies (`assumptions/stubs/P3BabyBear/Assumptions/FunsExternal.lean`) | no |

The `Debug::fmt` bodies are opaque by choice: `extract.sh` passes charon
`--opaque '{impl core::fmt::Debug for _}'`, because `core::fmt` is not modelled
and formatting has no bearing on the arithmetic.

## Layer 4 — the patches

### Pre-extraction: 3 patches

| Patch | What |
|---|---|
| `010-declare-hax-backend-lean-cfg` | declares the cfg name to rustc's check-cfg |
| `020-field-cfg-raw-data-serializable-supertrait` | under the cfg, `Field` stops implying `RawDataSerializable`; redundant bounds restated at 9 generic sites |
| `030-field-cfg-rpitit-packed-methods` | under the cfg, the four `-> impl Iterator` packed-trait methods are absent (`to_ext_iter` moves to an extension trait) |

`020` and `030` exist because of one limitation: rustc turns `-> impl Trait`
in a trait method into a hidden generic associated type, and charon/aeneas
cannot lift GATs (charon#1266). Without those two, aeneas reports 803 errors
and no usable output. `010` only declares the cfg name those two use, so
rustc's `unexpected_cfgs` lint stays quiet.

**What this costs.** The extracted crate is the cfg'd variant. Its items are
the shipped items minus the hidden ones: no body, constant or expression
differs. Two checks back that up:

1. *The normal build is unchanged.* `test-pre-patches.py` (step 2) runs
   `cargo test --workspace` on the tree as shipped and with the patches, and
   compares per test and per test source: **0 divergences
   (4671 tests)**. It writes the result into each patch's header (`# Tested:`
   and one `# Divergence:` entry per diverging test), and fails while any
   entry is unclassified. The only unconditional edits are implied bounds and
   the cfg declaration.
2. *Nothing reachable used what was hidden.* Step 3 compiles p3-baby-bear
   for `thumbv7em-none-eabi` under the cfg (the variant charon sees),
   warning-free.

The upstream run is 4671 tests: 4625 pass in both trees and 46 are
`#[ignore]`d in both.

### Charon scope flags (not patches, but they restrict what is extracted)

| Flag | Effect |
|---|---|
| `--targets thumbv7em-none-eabi` | scalar only: no SIMD module is compiled |
| `--opaque '{impl core::fmt::Debug for _}'` | `Debug` bodies become the four opaque constants above |
| `--exclude serde_core --exclude serde` and their impls | `Field: Serialize + DeserializeOwned` pulls in serde's mutually recursive `Serializer` family; serialisation is out of scope |

### Post-extraction: 3 patches

| Patch | Hunks | Cost |
|---|---:|---|
| `010-restore-dropped-default-methods` | 6 | none: adds `clone_from`/`ne` with the CoreModels default bodies |
| `020-trait-default-constant-shape` | 2 | none: calls three default constants with the signature their definition has |
| `030-name-anonymous-const-assertions` | 9 | none: renames `__N` → `const_check_N` |

## Layer 5 — the extractor: bugs found while building this

Each one is worked around above and worth reporting upstream.

1. **hax does not pass `-C --target` to charon** (`into lean`). Charon also
   ignores `CARGO_BUILD_TARGET`, since it always passes its own `--target
   <host>`. The only way through is charon's `--targets`, which leads to 2.
2. **charon `--targets` drops hax's default-method roots.** Multi-target mode
   forces `--translate-all-methods`, then filters unused methods after the
   merge, which removes `clone_from`/`ne` (11 → 2 in the LLBC). Post-patch 010.
3. **aeneas's defining and using runs disagree on default-constant shapes.**
   `N.default (Self : Type) : Usize` in p3-monty-31's run, called as
   `N.default <inst> : RustM Usize` from p3-baby-bear's. Post-patch 020.
4. **aeneas silently drops `MontyField31::new_array`/`new_2d_array`.** They
   are in the LLBC with bodies; no diagnostic. Hand-transcribed in
   `assumptions/Interface/P3Monty31Missing.lean`.
5. **aeneas's `__N` names for anonymous consts** trip mathlib's `nameCheck`
   linter. Post-patch 030.
6. **The seeded `Debug::fmt` signatures do not match `CoreModels`** (a
   `&mut Formatter` back-function aeneas adds but `core.fmt.Debug` lacks).
   Fixed in the hand-written `assumptions/stubs/P3BabyBear/Assumptions/FunsExternal.lean`.
7. **charon reports 2 `Type error after transformations` in the scoped
   p3-monty-31 run.** Both are in `core`'s own slice-iterator macros
   (`library/core/src/slice/iter/macros.rs:153`), which charon translates as a
   dependency, not in any Plonky3 item. aeneas exits 0 and nothing generated
   refers to them.
8. **charon splits option values on `,`**, so an impl pattern with two
   generic arguments cannot be written; the patterns omit the generics.
9. **Neither dependency extracts as a whole crate.** Whole-crate runs of
   p3-mds and p3-monty-31 (with the pre-patches) both fail in aeneas: each
   reaches p3-field's `PrimeCharacteristicRing`/`Algebra`/`Field`/`PrimeField`/
   `PackedField` group, which is mutually recursive through associated types
   ("mixed mutually recursive definitions"). `--opaque p3_field` does not help,
   because it hides bodies, not trait declarations; p3-mds also hits 28
   unsupported lifetime constraints of its own. Hence the scoped runs, and
   p3-field's traits in `assumptions/Interface/P3Field.lean`.

## Layer 6 — the specification: CompPoly and mathlib

CompPoly `v4.31.0` (mathlib `fabf563`, the same mathlib as aeneas):
`BabyBear.fieldSize`, `BabyBear.twoAdicity`, `BabyBear.is_prime` (a Pratt
certificate), `BabyBear.fieldSize_sub_one_factorization`.

## The theorems are not in the TCB

`spec/P3BabyBearProofs/` is proved, not assumed:

* `Constants.lean`: seven theorems about the constants
  (`monty_prime_eq_fieldSize`, `monty_prime_is_prime`, `two_adicity_eq_spec`,
  `monty_bits_eq_thirtyTwo`, `monty_mu_inverse`, `coprime_seven_pred_prime`,
  `fieldSize_sub_one_factorization'`), four wiring lemmas that the trait
  instances return those constants, and `two_adicity_maximal`
  (`2^(TWO_ADICITY+1) ∤ p - 1`). The factorization lemma fixes the odd part by
  hand, so on its own it does not say `TWO_ADICITY` is the *largest* such
  exponent; `two_adicity_maximal` does.
* `MontyField31.lean`: `new_spec`: `MontyField31::new` on BabyBear never
  panics (it discharges the four `const assert!`s aeneas keeps as `massert`s)
  and returns `x · 2^32 mod p`. Also `new_array_ok` and `new_2d_array_ok`,
  about the hand-written transcriptions, and the nine
  `poseidon{1,2}_const_check_N`. There are eleven length assertions. aeneas
  names the first in each module `_` and the rest `__N`; patch 030 renames
  the `__N` ones, and those nine are proved. `poseidon1._`
  (`BABYBEAR_POSEIDON1_RC_16`, `poseidon1.rs:66`) and `poseidon2._`
  (`BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL`, `poseidon2.rs:64`) are
  extracted and not proved. A single `_` does not trip `nameCheck`, which is
  why the patch does not rename them.

## What is *not* established

* Anything about field arithmetic, the Poseidon permutations, or the MDS
  layer. Those are extracted, or passed through abstract instances, but no
  theorem is about them.
* The two length assertions named `poseidon1._` and `poseidon2._` (above).
* Anything about the SIMD backends (see `../README.md`).
* That the hand-written transcriptions in `assumptions/` match upstream: read
  each against the Rust cited in its docstring.
