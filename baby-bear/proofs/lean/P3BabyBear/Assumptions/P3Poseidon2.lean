/-
# `p3-poseidon2`: hand-written interface

A scoped aeneas extraction of p3-poseidon2 translates the declarations below
but fails on the body of `Poseidon2::new` (see TCB.md, layer 3). The
structures are copied from that partial translation, so field names and order
are aeneas's own; `new` and `ExternalLayerConstants::new` are assumed.
-/
import P3Monty31.Assumptions.P3Field
open CoreModels Aeneas
open Aeneas.Std hiding namespace core alloc
open RustM ControlFlow Error

namespace p3_poseidon2

/-- [p3_poseidon2::external::ExternalLayerConstants].
Upstream: `poseidon2/src/external.rs:163`. -/
structure external.ExternalLayerConstants (T : Type) (WIDTH : Std.Usize) where
  initial : alloc.vec.Vec (Array T WIDTH)
  terminal : alloc.vec.Vec (Array T WIDTH)

/-- [p3_poseidon2::external::ExternalLayerConstants::new]. Upstream:
`poseidon2/src/external.rs:181`, a `const fn` that asserts
`initial.len() == terminal.len()` and builds the struct. Assumed, because a
body with that assertion would make the `ok`-ness of every caller depend on it. -/
opaque external.ExternalLayerConstants.new {T : Type} {WIDTH : Std.Usize}
    (initial : alloc.vec.Vec (Array T WIDTH))
    (terminal : alloc.vec.Vec (Array T WIDTH)) :
    RustM (external.ExternalLayerConstants T WIDTH)

/-- [p3_poseidon2::external::ExternalLayerConstructor].
Upstream: `poseidon2/src/external.rs:251`. -/
structure external.ExternalLayerConstructor (Self : Type) (F : Type)
    (WIDTH : Std.Usize) where
  p3_fieldfieldFieldInst : p3_field.field.Field F
  new_from_constants : external.ExternalLayerConstants F WIDTH → RustM Self

/-- [p3_poseidon2::internal::InternalLayerConstructor].
Upstream: `poseidon2/src/internal.rs:37`. -/
structure internal.InternalLayerConstructor (Self : Type) (F : Type) where
  p3_fieldfieldFieldInst : p3_field.field.Field F
  new_from_constants : alloc.vec.Vec F → RustM Self

/-- [p3_poseidon2::Poseidon2]. Upstream: `poseidon2/src/lib.rs:31`. -/
structure Poseidon2 (F : Type) (ExternalPerm : Type) (InternalPerm : Type)
    (WIDTH : Std.Usize) (D : Std.U64) where
  external_layer : ExternalPerm
  internal_layer : InternalPerm
  _phantom : core.marker.PhantomData F

/-- [p3_poseidon2::Poseidon2::new]. Upstream: `poseidon2/src/lib.rs:50`.
Assumed. -/
opaque Poseidon2.new {F ExternalPerm InternalPerm : Type}
    {WIDTH : Std.Usize} (D : Std.U64)
    (p3_fieldfieldPrimeFieldInst : p3_field.field.PrimeField F)
    (ExternalLayerConstructorInst :
      external.ExternalLayerConstructor ExternalPerm F WIDTH)
    (InternalLayerConstructorInst :
      internal.InternalLayerConstructor InternalPerm F)
    (external_constants : external.ExternalLayerConstants F WIDTH)
    (internal_constants : alloc.vec.Vec F) :
    RustM (Poseidon2 F ExternalPerm InternalPerm WIDTH D)

end p3_poseidon2
