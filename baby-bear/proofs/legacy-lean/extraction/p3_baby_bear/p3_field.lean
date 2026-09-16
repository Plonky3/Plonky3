import Hax

/-! Axiomatized fragment of the `p3-field` trait hierarchy that `p3-baby-bear`
    consumes. Only the classes, members and free functions the extracted
    `p3_baby_bear.lean` actually projects are declared; arity and the
    `AssociatedTypes` companion shape must match what hax emits. -/

class p3_field.field.PrimeCharacteristicRing.AssociatedTypes (R : Type) where

class p3_field.field.PrimeCharacteristicRing (R : Type)
  [associatedTypes : outParam
    (p3_field.field.PrimeCharacteristicRing.AssociatedTypes R)]
  where
  double (R) (val : R) : RustM R
  halve (R) (val : R) : RustM R
  div_2exp_u64 (R) (val : R) (exp : u64) : RustM R

-- copied from p3_field
namespace p3_field.field

class Algebra.AssociatedTypes (Self : Type) (F : Type) where
  [trait_constr_Algebra_i0 : PrimeCharacteristicRing.AssociatedTypes Self]

attribute [instance_reducible, instance]
  Algebra.AssociatedTypes.trait_constr_Algebra_i0

class Algebra (Self : Type) (F : Type)
  [associatedTypes : outParam (Algebra.AssociatedTypes (Self : Type) (F : Type))]
  where
  [trait_constr_Algebra_i0 : PrimeCharacteristicRing Self]

attribute [instance_reducible, instance] Algebra.trait_constr_Algebra_i0

/-- Rejection-sampling parameters. `p3-baby-bear` supplies both members, so they
    are declared with their real types and left without defaults. -/
class UniformSamplingField.AssociatedTypes (Self : Type) where

class UniformSamplingField (Self : Type)
  [associatedTypes : outParam (UniformSamplingField.AssociatedTypes
      (Self : Type))]
  where
  MAX_SINGLE_SAMPLE_BITS (Self) : usize
  SAMPLING_BITS_M (Self) : (RustArray u64 64)

end p3_field.field

-- copied from p3_field
namespace p3_field.dup

/-- `pub trait Dup: Clone { fn dup(&self) -> Self; }` with the blanket
    `impl<T: Copy + Clone> Dup for T { fn dup(&self) -> Self { *self } }`.
    The extraction calls `Dup.dup` without carrying a `Dup` bound, so the
    blanket instance has to be resolvable for any type. Its body `pure self`
    is the upstream body for `Copy` types, not an admission. -/
class Dup (Self : Type) where
  dup (Self) (self : Self) : RustM Self

instance (T : Type) : Dup T where
  dup := fun (self : T) => pure self

end p3_field.dup

namespace p3_field.exponentiation

/-- BabyBear's `exp_root_d`: `7 * 1725656503 = 6*(2^31 - 2^27) + 1 ≡ 1 mod (p-1)`.
    ASSUMED — the body here is the identity, which is *not* the upstream
    algorithm. Nothing may be concluded about the S-box from this stub. -/
def exp_1725656503
    (R : Type)
    [_trait_constr_exp_1725656503_associated_type_i0 :
      p3_field.field.PrimeCharacteristicRing.AssociatedTypes
      R]
    [_trait_constr_exp_1725656503_i0 : p3_field.field.PrimeCharacteristicRing R ]
    (val : R) :
    RustM R := pure val

end p3_field.exponentiation

/-! ### `PrimeCharacteristicRing`'s arithmetic supertraits

    Upstream declares

    ```rust
    pub trait PrimeCharacteristicRing:
        Sized + Default + Dup
        + Add<Output = Self> + AddAssign
        + Sub<Output = Self> + SubAssign
        + Neg<Output = Self>
        + Mul<Output = Self> + MulAssign
        + Sum + Product + Debug
    ```

    so every such ring carries `Add`/`Sub`/`Mul` with `Output = Self`. The
    extraction projects these inside generic functions, so they must be
    resolvable from the `PrimeCharacteristicRing` bound alone.

    The `Output := R` choices are FAITHFUL (they are what `Output = Self`
    means). The operation bodies are ADMITTED: this file models no ring
    arithmetic whatsoever. Each is guarded on the `PrimeCharacteristicRing`
    marker so it cannot collide with Hax's instances for the primitive
    integer types. -/

namespace p3_field.field

/-! ### The assumed ring operations

    These are the **core assumption** of this interface: no ring arithmetic is
    modelled. They are declared `opaque` rather than left as `sorry` for a
    concrete reason — `sorry` elaborates to `sorryAx`, which can inhabit
    `False`, so a `sorry`-bearing interface is formally inconsistent even though
    nothing exploits it. An `opaque` constant asserts only that *some* function
    of that type exists, which is true and is exactly what "axiomatized
    interface" is supposed to mean. It also keeps the build warning-clean, so
    a `sorry` introduced later by drift is immediately visible.

    `#print axioms` on anything downstream reports no axioms from these. That
    does NOT mean the operations are defined -- they are opaque. It means the
    assumption is well-formed rather than contradictory. -/

namespace p3_field.field

/-- ASSUMED: `PrimeCharacteristicRing`'s `Add` (`Output = Self`). -/
opaque ring_add (R : Type) : R -> R -> RustM R
/-- ASSUMED: `Sub` (`Output = Self`). -/
opaque ring_sub (R : Type) : R -> R -> RustM R
/-- ASSUMED: `Mul` (`Output = Self`). -/
opaque ring_mul (R : Type) : R -> R -> RustM R
/-- ASSUMED: `AddAssign`. -/
opaque ring_add_assign (R : Type) : R -> R -> RustM R
/-- ASSUMED: `SubAssign`. -/
opaque ring_sub_assign (R : Type) : R -> R -> RustM R
/-- ASSUMED: `Neg` (`Output = Self`). -/
opaque ring_neg (R : Type) : R -> RustM R

end p3_field.field

namespace p3_field.field

variable {R : Type}

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.Add.AssociatedTypes R R where
  Output := R
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.Add R R where
  add := ring_add R

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.Sub.AssociatedTypes R R where
  Output := R
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.Sub R R where
  sub := ring_sub R

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.Mul.AssociatedTypes R R where
  Output := R
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.Mul R R where
  mul := ring_mul R

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.AddAssign.AssociatedTypes R R where
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.AddAssign R R where
  add_assign := ring_add_assign R

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.SubAssign.AssociatedTypes R R where
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.SubAssign R R where
  sub_assign := ring_sub_assign R

@[reducible] instance [PrimeCharacteristicRing.AssociatedTypes R] :
  core_models.ops.arith.Neg.AssociatedTypes R where
  Output := R
instance [PrimeCharacteristicRing.AssociatedTypes R] [PrimeCharacteristicRing R] :
  core_models.ops.arith.Neg R where
  neg := ring_neg R

end p3_field.field
