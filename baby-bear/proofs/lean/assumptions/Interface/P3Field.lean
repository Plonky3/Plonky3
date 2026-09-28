/-
# `p3-field`: hand-written interface

aeneas cannot translate p3-field's trait hierarchy: `PrimeCharacteristicRing`,
`Algebra`, `Field`, `PrimeField` and `PackedField` are mutually recursive
through their associated types (`PrimeSubfield: PrimeField`,
`Packing: PackedField`), and a scoped extraction pulls in the traits' default
method bodies (`sqrt`, `tonelli_shanks`, …), which fail on their own. See
TCB.md, layer 3.

Every trait here is a Lean `structure`, as aeneas would emit one, but it declares
only the members that p3-baby-bear and the scoped dependency extractions
actually project. Instances are passed in as parameters, so anything proved
about code generic over one of these traits holds for *every* implementation,
which is weaker than assuming `opaque` ring operations. Field
names and argument order follow aeneas's naming scheme exactly, so the
generated code elaborates against them unchanged.
-/
import Aeneas
import CoreModels
open CoreModels Aeneas
open Aeneas.Std hiding namespace core alloc
open RustM ControlFlow Error

namespace p3_field

/-- [p3_field::dup::Dup]. Upstream: `field/src/dup.rs`, a blanket
`impl<T: Copy> Dup for T` whose body is `*self`. -/
structure dup.Dup (Self : Type) where
  dup : Self → RustM Self

/-- [p3_field::field::PrimeCharacteristicRing], restricted to the members the
extraction uses. Upstream: `field/src/field.rs:55`. -/
structure field.PrimeCharacteristicRing (Self : Type) where
  coreopsarithAddInst : core.ops.arith.Add Self Self Self
  coreopsarithSubInst : core.ops.arith.Sub Self Self Self
  coreopsarithAddAssignInst : core.ops.arith.AddAssign Self Self
  dupDupInst : dup.Dup Self
  double : Self → RustM Self
  halve : Self → RustM Self
  div_2exp_u64 : Self → Std.U64 → RustM Self

/-- [p3_field::field::Field], with no members: the extraction only passes
`Field` instances through (as the `p3_fieldfieldFieldInst` of the Poseidon
layer-constructor traits) and never projects from one. -/
structure field.Field (Self : Type) where

/-- [p3_field::field::PrimeField], with no members, for the same reason as
`Field`. Needed only as the `F: PrimeField` bound of `Poseidon1::new` and
`Poseidon2::new`. -/
structure field.PrimeField (Self : Type) where

/-- [p3_field::field::UniformSamplingField]. Upstream: `field/src/field.rs:613`.
Complete: the trait has exactly these two associated constants. -/
structure field.UniformSamplingField (Self : Type) where
  MAX_SINGLE_SAMPLE_BITS : RustM Std.Usize
  SAMPLING_BITS_M : RustM (Array Std.U64 64#usize)

/-- [p3_field::exponentiation::exp_1725656503]: `val ^ 1725656503`, via an
addition chain. Upstream: `field/src/exponentiation.rs:62`. Assumed: its body
is a chain of `square`/`mul` calls on a `PrimeCharacteristicRing` whose methods
are not modelled here. -/
opaque exponentiation.exp_1725656503 {R : Type}
    (PrimeCharacteristicRingInst : field.PrimeCharacteristicRing R) (val : R) :
    RustM R

end p3_field
