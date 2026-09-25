import Hax

/-! Gap-fillers for the **Hax proof library**, kept separate from the
    per-crate stubs so `TCB.md` can distinguish "we axiomatized an upstream
    Plonky3 crate" from "the Hax proof library is missing an instance".

    Entries here are candidates to upstream to cryspen/hax and delete. -/

/-- Blanket `AsRef` from a Rust array to its slice. Required by monty-31's
    `BinomialExtensionData.ArrayLike` bound, which fires when a concrete field
    instantiates the class. Hax declares the class but ships no such instance. -/
@[reducible] instance {T : Type} {N : usize} :
  core_models.convert.AsRef.AssociatedTypes (RustArray T N) (RustSlice T) where
instance {T : Type} {N : usize} :
  core_models.convert.AsRef (RustArray T N) (RustSlice T) where
  -- FAITHFUL: `RustSlice` is `rust_primitives.sequence.Seq`, and `unsize` is
  -- precisely the array-to-slice coercion. Not an assumption.
  as_ref := fun a => rust_primitives.unsize a
