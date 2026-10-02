/-
# `p3-poseidon1`: hand-written interface

A scoped aeneas extraction of p3-poseidon1 fails with an internal error in
`core::iter` (see TCB.md, layer 3), so these are written by hand after the
Rust declarations. p3-baby-bear only *constructs* Poseidon1 instances
(`default_babybear_poseidon1_{16,24}`), and nothing in `Verification/` is about the
permutation, so the types keep their upstream fields but `new` is assumed.
-/
import P3Monty31.Assumptions.P3Field
open CoreModels Aeneas
open Aeneas.Std hiding namespace core alloc
open RustM ControlFlow Error

namespace p3_poseidon1

/-- [p3_poseidon1::external::FullRoundConstants]. Its layout is not observed
by the extraction. -/
opaque external.FullRoundConstants (F : Type) (WIDTH : Std.Usize) : Type

/-- [p3_poseidon1::internal::PartialRoundConstants]. As above. -/
opaque internal.PartialRoundConstants (F : Type) (WIDTH : Std.Usize) : Type

/-- [p3_poseidon1::external::FullRoundLayerConstructor].
Upstream: `poseidon1/src/external.rs:68`. -/
structure external.FullRoundLayerConstructor (Self : Type) (F : Type)
    (WIDTH : Std.Usize) where
  p3_fieldfieldFieldInst : p3_field.field.Field F
  new_from_constants : external.FullRoundConstants F WIDTH → RustM Self

/-- [p3_poseidon1::internal::PartialRoundLayerConstructor].
Upstream: `poseidon1/src/internal.rs:115`. -/
structure internal.PartialRoundLayerConstructor (Self : Type) (F : Type)
    (WIDTH : Std.Usize) where
  p3_fieldfieldFieldInst : p3_field.field.Field F
  new_from_constants : internal.PartialRoundConstants F WIDTH → RustM Self

/-- [p3_poseidon1::Poseidon1Constants]. Upstream: `poseidon1/src/lib.rs:81`.
Complete: all four fields, in order. -/
structure Poseidon1Constants (F : Type) (WIDTH : Std.Usize) where
  rounds_f : Std.Usize
  rounds_p : Std.Usize
  mds_circ_col : Array Std.I64 WIDTH
  round_constants : alloc.vec.Vec (Array F WIDTH)

/-- [p3_poseidon1::Poseidon1]. Upstream: `poseidon1/src/lib.rs:167`.
Complete: all three fields, in order. -/
structure Poseidon1 (F : Type) (FullRoundPerm : Type) (PartialRoundPerm : Type)
    (WIDTH : Std.Usize) (D : Std.U64) where
  full_round_layer : FullRoundPerm
  partial_round_layer : PartialRoundPerm
  _phantom : core.marker.PhantomData F

/-- [p3_poseidon1::Poseidon1::new]. Upstream: `poseidon1/src/lib.rs:194`.
Assumed: the sparse-matrix decomposition in its body is not modelled. -/
opaque Poseidon1.new {F FullRoundPerm PartialRoundPerm : Type}
    {WIDTH : Std.Usize} (D : Std.U64)
    (p3_fieldfieldPrimeFieldInst : p3_field.field.PrimeField F)
    (FullRoundLayerConstructorInst :
      external.FullRoundLayerConstructor FullRoundPerm F WIDTH)
    (PartialRoundLayerConstructorInst :
      internal.PartialRoundLayerConstructor PartialRoundPerm F WIDTH)
    (raw : Poseidon1Constants F WIDTH) :
    RustM (Poseidon1 F FullRoundPerm PartialRoundPerm WIDTH D)

end p3_poseidon1
