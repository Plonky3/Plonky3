
-- Legacy lean backend for Hax
-- The Hax prelude library can be found in hax/proof-libs/legacy-lean
import Hax
import p3_baby_bear.dependencies -- PATCHED
import Std.Tactic.Do
import Std.Do.Triple
import Std.Tactic.Do.Syntax
open Std.Do
open Std.Tactic

set_option mvcgen.warning false
set_option linter.unusedVariables false
set_option maxRecDepth 8000 -- PATCHED: width-32 internal layer let-chain


namespace p3_baby_bear.baby_bear

structure BabyBearParameters where
  -- no fields

--  The prime field `2^31 - 2^27 + 1`, a.k.a. the Baby Bear field.
abbrev BabyBear -- PATCHED: instance binders hax does not propagate
  [ p3_monty_31.data_traits.MontyParameters.AssociatedTypes BabyBearParameters ]
  [ p3_monty_31.data_traits.MontyParameters BabyBearParameters ] :
  Type := (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
-- abbrev BabyBear : Type := (p3_monty_31.monty_31.MontyField31 BabyBearParameters)

@[instance] opaque Impl_11.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_11 :
  core_models.clone.Clone BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10.AssociatedTypes :
  core_models.marker.Copy.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_10 :
  core_models.marker.Copy BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12.AssociatedTypes :
  core_models.default.Default.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_12 :
  core_models.default.Default BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_13 :
  core_models.fmt.Debug BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_15.AssociatedTypes :
  core_models.hash.Hash.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_15 :
  core_models.hash.Hash BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_16.AssociatedTypes :
  core_models.marker.StructuralPartialEq.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_16 :
  core_models.marker.StructuralPartialEq BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_17.AssociatedTypes :
  core_models.cmp.PartialEq.AssociatedTypes
  BabyBearParameters
  BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_17 :
  core_models.cmp.PartialEq BabyBearParameters BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14.AssociatedTypes :
  core_models.cmp.Eq.AssociatedTypes BabyBearParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_14 :
  core_models.cmp.Eq BabyBearParameters :=
  by constructor <;> exact Inhabited.default

--  The Baby Bear prime: 2^31 - 2^27 + 1.
--  This is the unique 31-bit prime with the highest possible 2 adicity (27).
def Impl.PRIME_hoisted : u32 := (2013265921 : u32)

def Impl.MONTY_BITS_hoisted : u32 := (32 : u32)

def Impl.MONTY_MU_hoisted : u32 := (2281701377 : u32)

@[reducible] instance Impl.AssociatedTypes :
  p3_monty_31.data_traits.MontyParameters.AssociatedTypes BabyBearParameters
  where

instance Impl : p3_monty_31.data_traits.MontyParameters BabyBearParameters where
  PRIME := (Impl.PRIME_hoisted)
  MONTY_BITS := (Impl.MONTY_BITS_hoisted)
  MONTY_MU := (Impl.MONTY_MU_hoisted)

@[reducible] instance Impl_1.AssociatedTypes :
  p3_monty_31.data_traits.PackedMontyParameters.AssociatedTypes
  BabyBearParameters
  where

instance Impl_1 :
  p3_monty_31.data_traits.PackedMontyParameters BabyBearParameters
  where

@[reducible] instance Impl_2.AssociatedTypes :
  p3_monty_31.data_traits.BarrettParameters.AssociatedTypes BabyBearParameters
  where

instance Impl_2 :
  p3_monty_31.data_traits.BarrettParameters BabyBearParameters
  where

def Impl_3.MONTY_GEN_hoisted :
  (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
  :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (31 : u32)))
    (by rfl)

def Impl_3.BENEFITS_FROM_LOCKSTEP_EVALUATION_hoisted : Bool := true

@[reducible] instance Impl_3.AssociatedTypes :
  p3_monty_31.data_traits.FieldParameters.AssociatedTypes BabyBearParameters
  where

instance Impl_3 :
  p3_monty_31.data_traits.FieldParameters BabyBearParameters
  where
  MONTY_GEN := (Impl_3.MONTY_GEN_hoisted)
  BENEFITS_FROM_LOCKSTEP_EVALUATION :=
  (Impl_3.BENEFITS_FROM_LOCKSTEP_EVALUATION_hoisted)

--  In the field `BabyBear`, `a^{1/7}` is equal to a^{1725656503}.
-- 
--  This follows from the calculation `7 * 1725656503 = 6*(2^31 - 2^27) + 1 = 1 mod (p - 1)`.
@[spec]
def Impl_4.exp_root_d_hoisted
    (R : Type)
    [trait_constr_exp_root_d_hoisted_associated_type_i0 :
      p3_field.field.PrimeCharacteristicRing.AssociatedTypes
      R]
    [trait_constr_exp_root_d_hoisted_i0 : p3_field.field.PrimeCharacteristicRing
      R
      ]
    (val : R) :
    RustM R := do
  (p3_field.exponentiation.exp_1725656503 R val)

@[reducible] instance Impl_4.AssociatedTypes :
  p3_monty_31.data_traits.RelativelyPrimePower.AssociatedTypes
  BabyBearParameters
  ((7 : u64))
  where

instance Impl_4 :
  p3_monty_31.data_traits.RelativelyPrimePower BabyBearParameters ((7 : u64))
  where
  exp_root_d :=
    fun
      
      (R : Type)
      [trait_constr__associated_type_i0 :
        p3_field.field.PrimeCharacteristicRing.AssociatedTypes
        R]
      [trait_constr__i0 : p3_field.field.PrimeCharacteristicRing R ]
      =>
    (Impl_4.exp_root_d_hoisted R)

def Impl_5.TWO_ADICITY_hoisted : usize := (27 : usize)

def Impl_5.TWO_ADIC_GENERATORS_hoisted :
  (RustSlice (p3_monty_31.monty_31.MontyField31 BabyBearParameters))
  :=
  RustM.of_isOk
    (do
    (rust_primitives.unsize
      (← (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((28 : usize))
        (RustArray.ofVec #v[(1 : u32),
                              (2013265920 : u32),
                              (1728404513 : u32),
                              (1592366214 : u32),
                              (196396260 : u32),
                              (760005850 : u32),
                              (1721589904 : u32),
                              (397765732 : u32),
                              (1732600167 : u32),
                              (1753498361 : u32),
                              (341742893 : u32),
                              (1340477990 : u32),
                              (1282623253 : u32),
                              (298008106 : u32),
                              (1657000625 : u32),
                              (2009781145 : u32),
                              (1421947380 : u32),
                              (1286330022 : u32),
                              (1559589183 : u32),
                              (1049899240 : u32),
                              (195061667 : u32),
                              (414040701 : u32),
                              (570250684 : u32),
                              (1267047229 : u32),
                              (1003846038 : u32),
                              (1149491290 : u32),
                              (975630072 : u32),
                              (440564289 : u32)])))))
    (by rfl)

def Impl_5.ROOTS_8_hoisted :
  (RustSlice (p3_monty_31.monty_31.MontyField31 BabyBearParameters))
  :=
  RustM.of_isOk
    (do
    (rust_primitives.unsize
      (← (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((4 : usize))
        (RustArray.ofVec #v[(1 : u32),
                              (1592366214 : u32),
                              (1728404513 : u32),
                              (211723194 : u32)])))))
    (by rfl)

def Impl_5.INV_ROOTS_8_hoisted :
  (RustSlice (p3_monty_31.monty_31.MontyField31 BabyBearParameters))
  :=
  RustM.of_isOk
    (do
    (rust_primitives.unsize
      (← (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((4 : usize))
        (RustArray.ofVec #v[(1 : u32),
                              (1801542727 : u32),
                              (284861408 : u32),
                              (420899707 : u32)])))))
    (by rfl)

def Impl_5.ROOTS_16_hoisted :
  (RustSlice (p3_monty_31.monty_31.MontyField31 BabyBearParameters))
  :=
  RustM.of_isOk
    (do
    (rust_primitives.unsize
      (← (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((8 : usize))
        (RustArray.ofVec #v[(1 : u32),
                              (196396260 : u32),
                              (1592366214 : u32),
                              (78945800 : u32),
                              (1728404513 : u32),
                              (1400279418 : u32),
                              (211723194 : u32),
                              (1446056615 : u32)])))))
    (by rfl)

def Impl_5.INV_ROOTS_16_hoisted :
  (RustSlice (p3_monty_31.monty_31.MontyField31 BabyBearParameters))
  :=
  RustM.of_isOk
    (do
    (rust_primitives.unsize
      (← (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((8 : usize))
        (RustArray.ofVec #v[(1 : u32),
                              (567209306 : u32),
                              (1801542727 : u32),
                              (612986503 : u32),
                              (284861408 : u32),
                              (1934320121 : u32),
                              (420899707 : u32),
                              (1816869661 : u32)])))))
    (by rfl)

@[reducible] instance Impl_5.AssociatedTypes :
  p3_monty_31.data_traits.TwoAdicData.AssociatedTypes BabyBearParameters
  where
  ArrayLike := (RustSlice
  (p3_monty_31.monty_31.MontyField31 BabyBearParameters))

instance Impl_5 : p3_monty_31.data_traits.TwoAdicData BabyBearParameters where
  TWO_ADICITY := (Impl_5.TWO_ADICITY_hoisted)
  TWO_ADIC_GENERATORS := (Impl_5.TWO_ADIC_GENERATORS_hoisted)
  ROOTS_8 := (Impl_5.ROOTS_8_hoisted)
  INV_ROOTS_8 := (Impl_5.INV_ROOTS_8_hoisted)
  ROOTS_16 := (Impl_5.ROOTS_16_hoisted)
  INV_ROOTS_16 := (Impl_5.INV_ROOTS_16_hoisted)

def Impl_6.W_hoisted : (p3_monty_31.monty_31.MontyField31 BabyBearParameters) :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (11 : u32)))
    (by rfl)

def Impl_6.DTH_ROOT_hoisted :
  (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
  :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (1728404513 : u32)))
    (by rfl)

def Impl_6.EXT_GENERATOR_hoisted :
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((4 : usize))
      (RustArray.ofVec #v[(8 : u32), (1 : u32), (0 : u32), (0 : u32)])))
    (by rfl)

def Impl_6.EXT_TWO_ADICITY_hoisted : usize := (29 : usize)

def Impl_6.TWO_ADIC_EXTENSION_GENERATORS_hoisted :
  (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 4)
  2)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      BabyBearParameters
      ((4 : usize))
      ((2 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(0 : u32),
                                                (0 : u32),
                                                (1996171314 : u32),
                                                (0 : u32)]),
                            (RustArray.ofVec #v[(0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (124907976 : u32)])])))
    (by rfl)

@[reducible] instance Impl_6.AssociatedTypes :
  p3_monty_31.data_traits.BinomialExtensionData.AssociatedTypes
  BabyBearParameters
  ((4 : usize))
  where
  ArrayLike := (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 4)
  2)

instance Impl_6 :
  p3_monty_31.data_traits.BinomialExtensionData BabyBearParameters ((4 : usize))
  where
  W := (Impl_6.W_hoisted)
  DTH_ROOT := (Impl_6.DTH_ROOT_hoisted)
  EXT_GENERATOR := (Impl_6.EXT_GENERATOR_hoisted)
  EXT_TWO_ADICITY := (Impl_6.EXT_TWO_ADICITY_hoisted)
  TWO_ADIC_EXTENSION_GENERATORS :=
  (Impl_6.TWO_ADIC_EXTENSION_GENERATORS_hoisted)

def Impl_7.W_hoisted : (p3_monty_31.monty_31.MontyField31 BabyBearParameters) :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (2 : u32)))
    (by rfl)

@[spec]
def Impl_7.mul_w_hoisted
    (A : Type)
    [trait_constr_mul_w_hoisted_associated_type_i0 :
      p3_field.field.Algebra.AssociatedTypes
      A
      (p3_monty_31.monty_31.MontyField31 BabyBearParameters)]
    [trait_constr_mul_w_hoisted_i0 : p3_field.field.Algebra
      A
      (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
      ]
    (a : A) :
    RustM A := do
  (p3_field.field.PrimeCharacteristicRing.double A a)

def Impl_7.DTH_ROOT_hoisted :
  (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
  :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (815036133 : u32)))
    (by rfl)

def Impl_7.EXT_GENERATOR_hoisted :
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 5)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((5 : usize))
      (RustArray.ofVec #v[(8 : u32),
                            (1 : u32),
                            (0 : u32),
                            (0 : u32),
                            (0 : u32)])))
    (by rfl)

def Impl_7.EXT_TWO_ADICITY_hoisted : usize := (27 : usize)

def Impl_7.TWO_ADIC_EXTENSION_GENERATORS_hoisted :
  (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 5)
  0)
  :=
  RustM.of_isOk (do (pure (RustArray.ofVec #v[]))) (by rfl)

@[reducible] instance Impl_7.AssociatedTypes :
  p3_monty_31.data_traits.BinomialExtensionData.AssociatedTypes
  BabyBearParameters
  ((5 : usize))
  where
  ArrayLike := (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 5)
  0)

instance Impl_7 :
  p3_monty_31.data_traits.BinomialExtensionData BabyBearParameters ((5 : usize))
  where
  W := (Impl_7.W_hoisted)
  mul_w :=
    fun
      
      (A : Type)
      [trait_constr__associated_type_i0 : p3_field.field.Algebra.AssociatedTypes
        A
        (p3_monty_31.monty_31.MontyField31 BabyBearParameters)]
      [trait_constr__i0 : p3_field.field.Algebra
        A
        (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
        ]
      =>
    (Impl_7.mul_w_hoisted A)
  DTH_ROOT := (Impl_7.DTH_ROOT_hoisted)
  EXT_GENERATOR := (Impl_7.EXT_GENERATOR_hoisted)
  EXT_TWO_ADICITY := (Impl_7.EXT_TWO_ADICITY_hoisted)
  TWO_ADIC_EXTENSION_GENERATORS :=
  (Impl_7.TWO_ADIC_EXTENSION_GENERATORS_hoisted)

def Impl_8.W_hoisted : (p3_monty_31.monty_31.MontyField31 BabyBearParameters) :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (11 : u32)))
    (by rfl)

def Impl_8.DTH_ROOT_hoisted :
  (p3_monty_31.monty_31.MontyField31 BabyBearParameters)
  :=
  RustM.of_isOk
    (do (p3_monty_31.monty_31.Impl.new BabyBearParameters (420899707 : u32)))
    (by rfl)

def Impl_8.EXT_GENERATOR_hoisted :
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 8)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array BabyBearParameters ((8 : usize))
      (RustArray.ofVec #v[(5 : u32),
                            (1 : u32),
                            (0 : u32),
                            (0 : u32),
                            (0 : u32),
                            (0 : u32),
                            (0 : u32),
                            (0 : u32)])))
    (by rfl)

def Impl_8.EXT_TWO_ADICITY_hoisted : usize := (30 : usize)

def Impl_8.TWO_ADIC_EXTENSION_GENERATORS_hoisted :
  (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 8)
  3)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      BabyBearParameters
      ((8 : usize))
      ((3 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(0 : u32),
                                                (0 : u32),
                                                (0 : u32),
                                                (0 : u32),
                                                (1996171314 : u32),
                                                (0 : u32),
                                                (0 : u32),
                                                (0 : u32)]),
                            (RustArray.ofVec #v[(0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (124907976 : u32),
                                                  (0 : u32)]),
                            (RustArray.ofVec #v[(0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (518392818 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (0 : u32),
                                                  (0 : u32)])])))
    (by rfl)

@[reducible] instance Impl_8.AssociatedTypes :
  p3_monty_31.data_traits.BinomialExtensionData.AssociatedTypes
  BabyBearParameters
  ((8 : usize))
  where
  ArrayLike := (RustArray
  (RustArray (p3_monty_31.monty_31.MontyField31 BabyBearParameters) 8)
  3)

instance Impl_8 :
  p3_monty_31.data_traits.BinomialExtensionData BabyBearParameters ((8 : usize))
  where
  W := (Impl_8.W_hoisted)
  DTH_ROOT := (Impl_8.DTH_ROOT_hoisted)
  EXT_GENERATOR := (Impl_8.EXT_GENERATOR_hoisted)
  EXT_TWO_ADICITY := (Impl_8.EXT_TWO_ADICITY_hoisted)
  TWO_ADIC_EXTENSION_GENERATORS :=
  (Impl_8.TWO_ADIC_EXTENSION_GENERATORS_hoisted)

def Impl_9.MAX_SINGLE_SAMPLE_BITS_hoisted : usize := (27 : usize)

def Impl_9.SAMPLING_BITS_M_hoisted : (RustArray u64 64) :=
  RustM.of_isOk
    (do
    let prime : u64 ←
      (rust_primitives.hax.cast_op
        (p3_monty_31.data_traits.MontyParameters.PRIME BabyBearParameters) :
        RustM u64);
    let a : (RustArray u64 64) ←
      (rust_primitives.hax.repeat (0 : u64) (64 : usize));
    let k : usize := (0 : usize);
    let ⟨a, k⟩ ←
      (rust_primitives.hax.while_loop
        (fun ⟨a, k⟩ => (do (pure true) : RustM Bool))
        (fun ⟨a, k⟩ => (do (k <? (64 : usize)) : RustM Bool))
        (fun ⟨a, k⟩ =>
          (do
          (rust_primitives.hax.int.from_machine (0 : u32)) :
          RustM hax_lib.int.Int))
        (rust_primitives.hax.Tuple2.mk a k)
        (fun ⟨a, k⟩ =>
          (do
          let a : (RustArray u64 64) ←
            if (← (k ==? (0 : usize))) then do
              let a : (RustArray u64 64) ←
                (rust_primitives.hax.monomorphized_update_at.update_at_usize
                  a
                  k
                  prime);
              (pure a)
            else do
              let mask : u64 ← (~? (← ((← ((1 : u64) <<<? k)) -? (1 : u64))));
              let a : (RustArray u64 64) ←
                (rust_primitives.hax.monomorphized_update_at.update_at_usize
                  a
                  k
                  (← (prime &&&? mask)));
              (pure a);
          let k : usize ← (k +? (1 : usize));
          (pure (rust_primitives.hax.Tuple2.mk a k)) :
          RustM (rust_primitives.hax.Tuple2 (RustArray u64 64) usize))));
    (pure a))
    (by native_decide) -- PATCHED: `SAMPLING_BITS_M` is a Rust `const while`
    -- loop. hax compiles it to a monotone fixpoint the kernel cannot reduce,
    -- so `rfl` fails; it does *evaluate*, so `native_decide` discharges it,
    -- at the cost of the `Lean.ofReduceBool` axiom. See TCB.md layer 4.
    -- (by rfl)

@[reducible] instance Impl_9.AssociatedTypes :
  p3_field.field.UniformSamplingField.AssociatedTypes BabyBearParameters
  where

instance Impl_9 : p3_field.field.UniformSamplingField BabyBearParameters where
  MAX_SINGLE_SAMPLE_BITS := (Impl_9.MAX_SINGLE_SAMPLE_BITS_hoisted)
  SAMPLING_BITS_M := (Impl_9.SAMPLING_BITS_M_hoisted)

end p3_baby_bear.baby_bear


namespace p3_baby_bear.mds

structure MDSBabyBearData where
  -- no fields

@[instance] opaque Impl_1.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_1 :
  core_models.clone.Clone MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2.AssociatedTypes :
  core_models.default.Default.AssociatedTypes MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_2 :
  core_models.default.Default MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_3 :
  core_models.fmt.Debug MDSBabyBearData :=
  by constructor <;> exact Inhabited.default

def Impl.MATRIX_CIRC_MDS_8_COL_hoisted : (RustArray i64 8) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((8 : usize)) i64
      (RustArray.ofVec #v[(7 : i64),
                            (1 : i64),
                            (3 : i64),
                            (8 : i64),
                            (8 : i64),
                            (3 : i64),
                            (4 : i64),
                            (9 : i64)])))
    (by rfl)

def Impl.MATRIX_CIRC_MDS_12_COL_hoisted : (RustArray i64 12) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((12 : usize)) i64
      (RustArray.ofVec #v[(1 : i64),
                            (1 : i64),
                            (2 : i64),
                            (1 : i64),
                            (8 : i64),
                            (9 : i64),
                            (10 : i64),
                            (7 : i64),
                            (5 : i64),
                            (9 : i64),
                            (4 : i64),
                            (10 : i64)])))
    (by rfl)

def Impl.MATRIX_CIRC_MDS_16_COL_hoisted : (RustArray i64 16) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((16 : usize)) i64
      (RustArray.ofVec #v[(1 : i64),
                            (1 : i64),
                            (51 : i64),
                            (1 : i64),
                            (11 : i64),
                            (17 : i64),
                            (2 : i64),
                            (1 : i64),
                            (101 : i64),
                            (63 : i64),
                            (15 : i64),
                            (2 : i64),
                            (67 : i64),
                            (22 : i64),
                            (13 : i64),
                            (3 : i64)])))
    (by rfl)

def Impl.MATRIX_CIRC_MDS_24_COL_hoisted : (RustArray i64 24) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((24 : usize)) i64
      (RustArray.ofVec #v[(755673771 : i64),
                            (1686439191 : i64),
                            (401954077 : i64),
                            (82624181 : i64),
                            (1838262485 : i64),
                            (1617965094 : i64),
                            (416740298 : i64),
                            (1922433447 : i64),
                            (2009967074 : i64),
                            (1007636536 : i64),
                            (651504225 : i64),
                            (56639581 : i64),
                            (1761374664 : i64),
                            (613787421 : i64),
                            (1566027714 : i64),
                            (378133912 : i64),
                            (1009532350 : i64),
                            (203676737 : i64),
                            (86296562 : i64),
                            (1810161513 : i64),
                            (175003436 : i64),
                            (1551339770 : i64),
                            (400627958 : i64),
                            (142123135 : i64)])))
    (by rfl)

def Impl.MATRIX_CIRC_MDS_32_COL_hoisted : (RustArray i64 32) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((32 : usize)) i64
      (RustArray.ofVec #v[(197132288 : i64),
                            (736989057 : i64),
                            (863897170 : i64),
                            (1279604177 : i64),
                            (1257430066 : i64),
                            (766772495 : i64),
                            (1735032035 : i64),
                            (973518478 : i64),
                            (1586872753 : i64),
                            (744445088 : i64),
                            (590541117 : i64),
                            (1491556337 : i64),
                            (984014030 : i64),
                            (1618929200 : i64),
                            (1322614768 : i64),
                            (1846039899 : i64),
                            (1276466518 : i64),
                            (1484771534 : i64),
                            (1177490769 : i64),
                            (53101516 : i64),
                            (1580090346 : i64),
                            (1902156546 : i64),
                            (249134661 : i64),
                            (1445767145 : i64),
                            (750193437 : i64),
                            (1318329891 : i64),
                            (390750657 : i64),
                            (293245990 : i64),
                            (1935024439 : i64),
                            (1415143558 : i64),
                            (1216108043 : i64),
                            (1755622571 : i64)])))
    (by rfl)

def Impl.MATRIX_CIRC_MDS_64_COL_hoisted : (RustArray i64 64) :=
  RustM.of_isOk
    (do
    (p3_mds.util.first_row_to_first_col ((64 : usize)) i64
      (RustArray.ofVec #v[(962033528 : i64),
                            (7533793 : i64),
                            (186352644 : i64),
                            (69115016 : i64),
                            (852635551 : i64),
                            (1313601862 : i64),
                            (549959383 : i64),
                            (92783207 : i64),
                            (1434939945 : i64),
                            (1293446852 : i64),
                            (1243217187 : i64),
                            (442802703 : i64),
                            (694536781 : i64),
                            (383355482 : i64),
                            (1193169994 : i64),
                            (423451787 : i64),
                            (1852235532 : i64),
                            (683613974 : i64),
                            (651889608 : i64),
                            (1527172571 : i64),
                            (269339810 : i64),
                            (1707239597 : i64),
                            (429853159 : i64),
                            (918160574 : i64),
                            (113480690 : i64),
                            (1370516989 : i64),
                            (1766961348 : i64),
                            (1719903426 : i64),
                            (995348763 : i64),
                            (652546892 : i64),
                            (1168503237 : i64),
                            (1867326515 : i64),
                            (1021499438 : i64),
                            (909154941 : i64),
                            (1037677657 : i64),
                            (1745165900 : i64),
                            (1547347744 : i64),
                            (291767575 : i64),
                            (104126869 : i64),
                            (453993478 : i64),
                            (1923187760 : i64),
                            (642858181 : i64),
                            (1740683580 : i64),
                            (1418376355 : i64),
                            (1670763021 : i64),
                            (1046330433 : i64),
                            (858453552 : i64),
                            (1012784307 : i64),
                            (178870162 : i64),
                            (1726698642 : i64),
                            (1064885724 : i64),
                            (1522800167 : i64),
                            (382554404 : i64),
                            (895802006 : i64),
                            (1176004553 : i64),
                            (360402449 : i64),
                            (446990024 : i64),
                            (206607785 : i64),
                            (942915782 : i64),
                            (9431973 : i64),
                            (1391246894 : i64),
                            (488082023 : i64),
                            (1488175179 : i64),
                            (1855310673 : i64)])))
    (by rfl)

@[reducible] instance Impl.AssociatedTypes :
  p3_monty_31.mds.MDSUtils.AssociatedTypes MDSBabyBearData
  where

instance Impl : p3_monty_31.mds.MDSUtils MDSBabyBearData where
  MATRIX_CIRC_MDS_8_COL := (Impl.MATRIX_CIRC_MDS_8_COL_hoisted)
  MATRIX_CIRC_MDS_12_COL := (Impl.MATRIX_CIRC_MDS_12_COL_hoisted)
  MATRIX_CIRC_MDS_16_COL := (Impl.MATRIX_CIRC_MDS_16_COL_hoisted)
  MATRIX_CIRC_MDS_24_COL := (Impl.MATRIX_CIRC_MDS_24_COL_hoisted)
  MATRIX_CIRC_MDS_32_COL := (Impl.MATRIX_CIRC_MDS_32_COL_hoisted)
  MATRIX_CIRC_MDS_64_COL := (Impl.MATRIX_CIRC_MDS_64_COL_hoisted)

abbrev MdsMatrixBabyBear :
  Type :=
  (p3_monty_31.mds.MdsMatrixMontyField31 MDSBabyBearData)

end p3_baby_bear.mds


namespace p3_baby_bear.poseidon1

--  External (full round) layer for BabyBear Poseidon1.
abbrev Poseidon1ExternalLayerBabyBear (WIDTH : usize) :
  Type :=
  (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    p3_baby_bear.mds.MDSBabyBearData
    (WIDTH))

--  S-box degree for BabyBear Poseidon1.
-- 
--  Since `p - 1 = 15 * 2^27`, both 3 and 5 divide `p - 1`.
-- 
--  So `gcd(α, p - 1) ≠ 1` for `α ∈ {3, 5}`. The next smallest valid exponent is 7.
def BABYBEAR_S_BOX_DEGREE : u64 := (7 : u64)

--  Number of full rounds per half for BabyBear Poseidon1 (`RF / 2`).
-- 
--  The total number of full rounds is `RF = 8` (4 beginning + 4 ending).
--  Follows the Poseidon1 paper's security analysis (Section 5.4) with a +2 RF margin.
def BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS : usize := (4 : usize)

--  Number of partial rounds for BabyBear Poseidon1 (width 16).
-- 
--  Derived from the Gröbner basis bound in the Poseidon1 paper (Eq. 4, line 2)
--  and the Poseidon2 paper (Eq. 1, R_GB term 3):
-- 
--    R_GB ≥ t − 7 + log_α(2) · min{κ/(t+1), log_2(p)/2}
--         = 9 + 0.3562 · min{7.53, 15.5} = 11.682
-- 
--  With the +7.5% security margin (Section 5.4): ⌈1.075 × 11.682⌉ = 13.
def BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16 : usize := (13 : usize)

--  Number of partial rounds for BabyBear Poseidon1 (width 24).
-- 
--  Same Gröbner basis bound as width 16:
-- 
--    R_GB ≥ 17 + 0.3562 · min{5.12, 15.5} = 18.824
-- 
--  With the +7.5% security margin: ⌈1.075 × 18.824⌉ = 21.
def BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24 : usize := (21 : usize)

--  Parameters for the Poseidon1 internal layer on BabyBear.
structure BabyBearPoseidonParameters where
  -- no fields

--  Internal (partial round) layer for BabyBear Poseidon1.
abbrev Poseidon1InternalLayerBabyBear (WIDTH : usize) :
  Type :=
  (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    (WIDTH)
    BabyBearPoseidonParameters)

--  The Poseidon1 permutation for BabyBear.
-- 
--  Acts on arrays of the form `[BabyBear; WIDTH]` or `[BabyBear::Packing; WIDTH]`.
abbrev Poseidon1BabyBear (WIDTH : usize) :
  Type :=
  (p3_poseidon1.Poseidon1
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      p3_baby_bear.mds.MDSBabyBearData
      (WIDTH))
    (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      (WIDTH)
      BabyBearPoseidonParameters)
    (WIDTH)
    ((7 : u64)))

--  Generic Poseidon1 linear layers for BabyBear.
-- 
--  Can act on `[A; WIDTH]` for any ring implementing `Algebra<BabyBear>`.
abbrev GenericPoseidon1LinearLayersBabyBear :
  Type :=
  (p3_monty_31.poseidon1.GenericPoseidon1LinearLayersMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    BabyBearPoseidonParameters)

@[instance] opaque Impl_4.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_4 :
  core_models.fmt.Debug BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_5.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_5 :
  core_models.clone.Clone BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6.AssociatedTypes :
  core_models.default.Default.AssociatedTypes BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6 :
  core_models.default.Default BabyBearPoseidonParameters :=
  by constructor <;> exact Inhabited.default

def Impl.USE_TEXTBOOK_hoisted : Bool := true

@[spec]
def Impl.mds_permute_hoisted
    (state :
    (RustArray
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    16)) :
    RustM
    (RustArray
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    16)
    := do
  let
    state : (RustArray
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    16) ←
    (p3_symmetric.permutation.Permutation.permute_mut
      (p3_monty_31.mds.MdsMatrixMontyField31 p3_baby_bear.mds.MDSBabyBearData)
      (RustArray
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      16)
      (← (core_models.default.Default.default
        (p3_monty_31.mds.MdsMatrixMontyField31 p3_baby_bear.mds.MDSBabyBearData)
        rust_primitives.hax.Tuple0.mk))
      state);
  (pure state)

@[reducible] instance Impl.AssociatedTypes :
  p3_monty_31.poseidon1.PartialRoundBaseParameters.AssociatedTypes
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

instance Impl :
  p3_monty_31.poseidon1.PartialRoundBaseParameters
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where
  USE_TEXTBOOK := (Impl.USE_TEXTBOOK_hoisted)
  mds_permute := (Impl.mds_permute_hoisted)

@[reducible] instance Impl_1.AssociatedTypes :
  p3_monty_31.poseidon1.PartialRoundBaseParameters.AssociatedTypes
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

instance Impl_1 :
  p3_monty_31.poseidon1.PartialRoundBaseParameters
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

@[reducible] instance Impl_2.AssociatedTypes :
  p3_monty_31.poseidon1.PartialRoundParameters.AssociatedTypes
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

instance Impl_2 :
  p3_monty_31.poseidon1.PartialRoundParameters
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

@[reducible] instance Impl_3.AssociatedTypes :
  p3_monty_31.poseidon1.PartialRoundParameters.AssociatedTypes
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

instance Impl_3 :
  p3_monty_31.poseidon1.PartialRoundParameters
  BabyBearPoseidonParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

--  Round constants for width-16 Poseidon1 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=16, R_F=8, R_P=13
-- 
--  Generated by `poseidon/generate_constants.py --field babybear --width 16`.
-- 
--  Layout: [initial_full (4 rounds), partial (13 rounds), terminal_full (4 rounds)].
def BABYBEAR_POSEIDON1_RC_16 :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  16)
  21)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize))
      ((21 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(1774958255 : u32),
                                                (1185780729 : u32),
                                                (1621102414 : u32),
                                                (1796380621 : u32),
                                                (588815102 : u32),
                                                (1932426223 : u32),
                                                (1925334750 : u32),
                                                (747903232 : u32),
                                                (89648862 : u32),
                                                (360728943 : u32),
                                                (977184635 : u32),
                                                (1425273457 : u32),
                                                (256487465 : u32),
                                                (1200041953 : u32),
                                                (572403254 : u32),
                                                (448208942 : u32)]),
                            (RustArray.ofVec #v[(1215789478 : u32),
                                                  (944884184 : u32),
                                                  (953948096 : u32),
                                                  (547326025 : u32),
                                                  (646827752 : u32),
                                                  (889997530 : u32),
                                                  (1536873262 : u32),
                                                  (86189867 : u32),
                                                  (1065944411 : u32),
                                                  (32019634 : u32),
                                                  (333311454 : u32),
                                                  (456061748 : u32),
                                                  (1963448500 : u32),
                                                  (1827584334 : u32),
                                                  (1391160226 : u32),
                                                  (1348741381 : u32)]),
                            (RustArray.ofVec #v[(88424255 : u32),
                                                  (104111868 : u32),
                                                  (1763866748 : u32),
                                                  (79691676 : u32),
                                                  (1988915530 : u32),
                                                  (1050669594 : u32),
                                                  (359890076 : u32),
                                                  (573163527 : u32),
                                                  (222820492 : u32),
                                                  (159256268 : u32),
                                                  (669703072 : u32),
                                                  (763177444 : u32),
                                                  (889367200 : u32),
                                                  (256335831 : u32),
                                                  (704371273 : u32),
                                                  (25886717 : u32)]),
                            (RustArray.ofVec #v[(51754520 : u32),
                                                  (1833211857 : u32),
                                                  (454499742 : u32),
                                                  (1384520381 : u32),
                                                  (777848065 : u32),
                                                  (1053320300 : u32),
                                                  (1851729162 : u32),
                                                  (344647910 : u32),
                                                  (401996362 : u32),
                                                  (1046925956 : u32),
                                                  (5351995 : u32),
                                                  (1212119315 : u32),
                                                  (754867989 : u32),
                                                  (36972490 : u32),
                                                  (751272725 : u32),
                                                  (506915399 : u32)]),
                            (RustArray.ofVec #v[(1518359488 : u32),
                                                  (1765533241 : u32),
                                                  (945325693 : u32),
                                                  (422793067 : u32),
                                                  (311365592 : u32),
                                                  (1311448267 : u32),
                                                  (1629555936 : u32),
                                                  (1009879353 : u32),
                                                  (190525218 : u32),
                                                  (786108885 : u32),
                                                  (557776863 : u32),
                                                  (212616710 : u32),
                                                  (605745517 : u32),
                                                  (1922082829 : u32),
                                                  (1870549801 : u32),
                                                  (1502529704 : u32)]),
                            (RustArray.ofVec #v[(1990744480 : u32),
                                                  (1700391016 : u32),
                                                  (1702593455 : u32),
                                                  (321330495 : u32),
                                                  (528965731 : u32),
                                                  (183414327 : u32),
                                                  (1886297254 : u32),
                                                  (1178602734 : u32),
                                                  (1923111974 : u32),
                                                  (744004766 : u32),
                                                  (549271463 : u32),
                                                  (1781349648 : u32),
                                                  (542259047 : u32),
                                                  (1536158148 : u32),
                                                  (715456982 : u32),
                                                  (503426110 : u32)]),
                            (RustArray.ofVec #v[(340311124 : u32),
                                                  (1558555932 : u32),
                                                  (1226350925 : u32),
                                                  (742828095 : u32),
                                                  (1338992758 : u32),
                                                  (1641600456 : u32),
                                                  (1843351545 : u32),
                                                  (301835475 : u32),
                                                  (43203215 : u32),
                                                  (386838401 : u32),
                                                  (1520185679 : u32),
                                                  (1235297680 : u32),
                                                  (904680097 : u32),
                                                  (1491801617 : u32),
                                                  (1581784677 : u32),
                                                  (913384905 : u32)]),
                            (RustArray.ofVec #v[(247083962 : u32),
                                                  (532844013 : u32),
                                                  (107190701 : u32),
                                                  (213827818 : u32),
                                                  (1979521776 : u32),
                                                  (1358282574 : u32),
                                                  (1681743681 : u32),
                                                  (1867507480 : u32),
                                                  (1530706910 : u32),
                                                  (507181886 : u32),
                                                  (695185447 : u32),
                                                  (1172395131 : u32),
                                                  (1250800299 : u32),
                                                  (1503161625 : u32),
                                                  (817684387 : u32),
                                                  (498481458 : u32)]),
                            (RustArray.ofVec #v[(494676004 : u32),
                                                  (1404253825 : u32),
                                                  (108246855 : u32),
                                                  (59414691 : u32),
                                                  (744214112 : u32),
                                                  (890862029 : u32),
                                                  (1342765939 : u32),
                                                  (1417398904 : u32),
                                                  (1897591937 : u32),
                                                  (1066647396 : u32),
                                                  (1682806907 : u32),
                                                  (1015795079 : u32),
                                                  (1619482808 : u32),
                                                  (199831866 : u32),
                                                  (559491384 : u32),
                                                  (1832496133 : u32)]),
                            (RustArray.ofVec #v[(481896934 : u32),
                                                  (1963254086 : u32),
                                                  (96164646 : u32),
                                                  (437790241 : u32),
                                                  (19508407 : u32),
                                                  (261251027 : u32),
                                                  (1633410288 : u32),
                                                  (1292273321 : u32),
                                                  (265700364 : u32),
                                                  (711884039 : u32),
                                                  (336503636 : u32),
                                                  (115823818 : u32),
                                                  (170978685 : u32),
                                                  (299219445 : u32),
                                                  (1608043399 : u32),
                                                  (887466445 : u32)]),
                            (RustArray.ofVec #v[(383594057 : u32),
                                                  (1150607499 : u32),
                                                  (601394258 : u32),
                                                  (973684106 : u32),
                                                  (1076545860 : u32),
                                                  (1178878502 : u32),
                                                  (1841426339 : u32),
                                                  (794989665 : u32),
                                                  (2011231942 : u32),
                                                  (563217520 : u32),
                                                  (595969490 : u32),
                                                  (1208186053 : u32),
                                                  (1599541087 : u32),
                                                  (831398504 : u32),
                                                  (264437180 : u32),
                                                  (1430033642 : u32)]),
                            (RustArray.ofVec #v[(227454140 : u32),
                                                  (1136308868 : u32),
                                                  (1336491500 : u32),
                                                  (1340151003 : u32),
                                                  (1701644979 : u32),
                                                  (125169733 : u32),
                                                  (1587988100 : u32),
                                                  (83379324 : u32),
                                                  (1761351423 : u32),
                                                  (1041104368 : u32),
                                                  (797580548 : u32),
                                                  (1643353372 : u32),
                                                  (798149378 : u32),
                                                  (542018779 : u32),
                                                  (1752997855 : u32),
                                                  (806408342 : u32)]),
                            (RustArray.ofVec #v[(667524710 : u32),
                                                  (546850713 : u32),
                                                  (1496608654 : u32),
                                                  (1991350329 : u32),
                                                  (1317630228 : u32),
                                                  (854423119 : u32),
                                                  (158597540 : u32),
                                                  (62186516 : u32),
                                                  (251077054 : u32),
                                                  (344805683 : u32),
                                                  (409989363 : u32),
                                                  (1183034575 : u32),
                                                  (888423197 : u32),
                                                  (1016270808 : u32),
                                                  (1357767266 : u32),
                                                  (944008412 : u32)]),
                            (RustArray.ofVec #v[(1259573942 : u32),
                                                  (671790555 : u32),
                                                  (166219691 : u32),
                                                  (1272738107 : u32),
                                                  (138387470 : u32),
                                                  (1468574072 : u32),
                                                  (1750288852 : u32),
                                                  (497840985 : u32),
                                                  (1537125420 : u32),
                                                  (1927291466 : u32),
                                                  (1929142309 : u32),
                                                  (174148619 : u32),
                                                  (1374113779 : u32),
                                                  (957716665 : u32),
                                                  (743716138 : u32),
                                                  (1349134576 : u32)]),
                            (RustArray.ofVec #v[(47205229 : u32),
                                                  (1514372976 : u32),
                                                  (1423561577 : u32),
                                                  (1545537398 : u32),
                                                  (1626693313 : u32),
                                                  (403634072 : u32),
                                                  (1455319009 : u32),
                                                  (1746651388 : u32),
                                                  (1705927622 : u32),
                                                  (604680790 : u32),
                                                  (1046653674 : u32),
                                                  (1176652533 : u32),
                                                  (1320084208 : u32),
                                                  (201946506 : u32),
                                                  (313095126 : u32),
                                                  (269393902 : u32)]),
                            (RustArray.ofVec #v[(494719588 : u32),
                                                  (1031680552 : u32),
                                                  (630851086 : u32),
                                                  (122326311 : u32),
                                                  (1231160851 : u32),
                                                  (1391127112 : u32),
                                                  (1673152090 : u32),
                                                  (9195380 : u32),
                                                  (1640379595 : u32),
                                                  (764983014 : u32),
                                                  (624674469 : u32),
                                                  (1015319981 : u32),
                                                  (1409894890 : u32),
                                                  (1334193631 : u32),
                                                  (1604926013 : u32),
                                                  (1550162856 : u32)]),
                            (RustArray.ofVec #v[(260778324 : u32),
                                                  (1522798728 : u32),
                                                  (1996894673 : u32),
                                                  (905323278 : u32),
                                                  (89038071 : u32),
                                                  (1961803864 : u32),
                                                  (1283653694 : u32),
                                                  (1460526778 : u32),
                                                  (1818122570 : u32),
                                                  (886011289 : u32),
                                                  (246824626 : u32),
                                                  (1899900378 : u32),
                                                  (1851364380 : u32),
                                                  (426802027 : u32),
                                                  (311703535 : u32),
                                                  (1472518969 : u32)]),
                            (RustArray.ofVec #v[(913094636 : u32),
                                                  (241382878 : u32),
                                                  (1578333386 : u32),
                                                  (1228732114 : u32),
                                                  (805189379 : u32),
                                                  (433929041 : u32),
                                                  (1029998479 : u32),
                                                  (1056194280 : u32),
                                                  (1685208700 : u32),
                                                  (1340424586 : u32),
                                                  (259745565 : u32),
                                                  (1060821900 : u32),
                                                  (405863817 : u32),
                                                  (1528116846 : u32),
                                                  (1017716813 : u32),
                                                  (345828924 : u32)]),
                            (RustArray.ofVec #v[(1417394521 : u32),
                                                  (1298810016 : u32),
                                                  (1997935306 : u32),
                                                  (1062984998 : u32),
                                                  (597940612 : u32),
                                                  (386598845 : u32),
                                                  (1512722580 : u32),
                                                  (1853778100 : u32),
                                                  (524059424 : u32),
                                                  (268155062 : u32),
                                                  (1589539906 : u32),
                                                  (1174390528 : u32),
                                                  (1691482038 : u32),
                                                  (456177249 : u32),
                                                  (1483646208 : u32),
                                                  (1345393149 : u32)]),
                            (RustArray.ofVec #v[(1924406519 : u32),
                                                  (1684581733 : u32),
                                                  (1690768305 : u32),
                                                  (1158356173 : u32),
                                                  (555638824 : u32),
                                                  (806031788 : u32),
                                                  (1948411476 : u32),
                                                  (1446021700 : u32),
                                                  (1520786250 : u32),
                                                  (1391639706 : u32),
                                                  (436694608 : u32),
                                                  (1589192852 : u32),
                                                  (331001711 : u32),
                                                  (1293530661 : u32),
                                                  (819090071 : u32),
                                                  (1391463443 : u32)]),
                            (RustArray.ofVec #v[(711414743 : u32),
                                                  (1086183646 : u32),
                                                  (581745560 : u32),
                                                  (2011436628 : u32),
                                                  (598436623 : u32),
                                                  (103409622 : u32),
                                                  (962047 : u32),
                                                  (1691127425 : u32),
                                                  (911196178 : u32),
                                                  (1137461453 : u32),
                                                  (1629204265 : u32),
                                                  (1220409384 : u32),
                                                  (445296751 : u32),
                                                  (288286176 : u32),
                                                  (57541084 : u32),
                                                  (403727924 : u32)])])))
    (by rfl)

def const_check_1 : rust_primitives.hax.Tuple0 := -- PATCHED: hax emitted `def _`
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          16) (← (rust_primitives.unsize BABYBEAR_POSEIDON1_RC_16))))
        ==? (← ((← ((2 : usize) *? BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS))
          +? BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16))))))
    (by rfl)

--  Round constants for width-24 Poseidon1 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=24, R_F=8, R_P=21
-- 
--  Generated by `poseidon/generate_constants.py --field babybear --width 24`.
-- 
--  Layout: [initial_full (4 rounds), partial (21 rounds), terminal_full (4 rounds)].
def BABYBEAR_POSEIDON1_RC_24 :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  24)
  29)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize))
      ((29 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(262278199 : u32),
                                                (127253399 : u32),
                                                (314968988 : u32),
                                                (246143118 : u32),
                                                (157582794 : u32),
                                                (118043943 : u32),
                                                (454905424 : u32),
                                                (815798990 : u32),
                                                (1004040026 : u32),
                                                (1773108264 : u32),
                                                (1066694495 : u32),
                                                (1930780904 : u32),
                                                (1180307149 : u32),
                                                (1464793095 : u32),
                                                (1660766320 : u32),
                                                (1389166148 : u32),
                                                (343354132 : u32),
                                                (1307439985 : u32),
                                                (638242172 : u32),
                                                (525458520 : u32),
                                                (1964135730 : u32),
                                                (1751797115 : u32),
                                                (1421525369 : u32),
                                                (831813382 : u32)]),
                            (RustArray.ofVec #v[(695835963 : u32),
                                                  (1845603984 : u32),
                                                  (540703332 : u32),
                                                  (1333667262 : u32),
                                                  (1917861751 : u32),
                                                  (1170029417 : u32),
                                                  (1989924532 : u32),
                                                  (1518763784 : u32),
                                                  (1339793538 : u32),
                                                  (622609176 : u32),
                                                  (686842369 : u32),
                                                  (1737016378 : u32),
                                                  (1282239129 : u32),
                                                  (897025192 : u32),
                                                  (716894289 : u32),
                                                  (1997503974 : u32),
                                                  (395622276 : u32),
                                                  (1201063290 : u32),
                                                  (1917549072 : u32),
                                                  (1150912935 : u32),
                                                  (1687379185 : u32),
                                                  (1507936940 : u32),
                                                  (241306552 : u32),
                                                  (989176635 : u32)]),
                            (RustArray.ofVec #v[(1147522062 : u32),
                                                  (27129487 : u32),
                                                  (1257820264 : u32),
                                                  (142102402 : u32),
                                                  (217046702 : u32),
                                                  (1664590951 : u32),
                                                  (855276054 : u32),
                                                  (1215259350 : u32),
                                                  (946500736 : u32),
                                                  (552696906 : u32),
                                                  (1424297384 : u32),
                                                  (538103555 : u32),
                                                  (1608853840 : u32),
                                                  (162510541 : u32),
                                                  (623051854 : u32),
                                                  (1549062383 : u32),
                                                  (1908416316 : u32),
                                                  (1622328571 : u32),
                                                  (1079030649 : u32),
                                                  (1584033957 : u32),
                                                  (1099252725 : u32),
                                                  (1910423126 : u32),
                                                  (447555988 : u32),
                                                  (862495875 : u32)]),
                            (RustArray.ofVec #v[(128479034 : u32),
                                                  (1587822577 : u32),
                                                  (608401422 : u32),
                                                  (1290028279 : u32),
                                                  (342857858 : u32),
                                                  (825405577 : u32),
                                                  (427731030 : u32),
                                                  (1718628547 : u32),
                                                  (588764636 : u32),
                                                  (204228775 : u32),
                                                  (1454563174 : u32),
                                                  (1740472809 : u32),
                                                  (1338899225 : u32),
                                                  (1269493554 : u32),
                                                  (53007114 : u32),
                                                  (1647670797 : u32),
                                                  (306391314 : u32),
                                                  (172614232 : u32),
                                                  (51256176 : u32),
                                                  (1221257987 : u32),
                                                  (1239734761 : u32),
                                                  (273790406 : u32),
                                                  (1781980094 : u32),
                                                  (1291790245 : u32)]),
                            (RustArray.ofVec #v[(497520322 : u32),
                                                  (1930103076 : u32),
                                                  (1052077299 : u32),
                                                  (1540960371 : u32),
                                                  (924863639 : u32),
                                                  (1365519753 : u32),
                                                  (1726563304 : u32),
                                                  (440300254 : u32),
                                                  (1891545577 : u32),
                                                  (822033215 : u32),
                                                  (1111544260 : u32),
                                                  (308575117 : u32),
                                                  (1708681573 : u32),
                                                  (1240419708 : u32),
                                                  (1199068823 : u32),
                                                  (1186174623 : u32),
                                                  (1551596046 : u32),
                                                  (1886977120 : u32),
                                                  (1327682690 : u32),
                                                  (1210751726 : u32),
                                                  (1810596765 : u32),
                                                  (53041581 : u32),
                                                  (723038058 : u32),
                                                  (1439947916 : u32)]),
                            (RustArray.ofVec #v[(1136469704 : u32),
                                                  (205609311 : u32),
                                                  (1883820770 : u32),
                                                  (14387587 : u32),
                                                  (720724951 : u32),
                                                  (1854174607 : u32),
                                                  (1629316321 : u32),
                                                  (530151394 : u32),
                                                  (1679178250 : u32),
                                                  (1549779579 : u32),
                                                  (48375137 : u32),
                                                  (976057819 : u32),
                                                  (463976218 : u32),
                                                  (875839332 : u32),
                                                  (1946596189 : u32),
                                                  (434078361 : u32),
                                                  (1878280202 : u32),
                                                  (1363837384 : u32),
                                                  (1470845646 : u32),
                                                  (1792450386 : u32),
                                                  (1040977421 : u32),
                                                  (1209164052 : u32),
                                                  (714957516 : u32),
                                                  (390340387 : u32)]),
                            (RustArray.ofVec #v[(1213686459 : u32),
                                                  (790726260 : u32),
                                                  (117294666 : u32),
                                                  (140621810 : u32),
                                                  (993455846 : u32),
                                                  (1889603648 : u32),
                                                  (78845751 : u32),
                                                  (925018226 : u32),
                                                  (708123747 : u32),
                                                  (1647665372 : u32),
                                                  (1649953458 : u32),
                                                  (942439428 : u32),
                                                  (1006235079 : u32),
                                                  (238616145 : u32),
                                                  (930036496 : u32),
                                                  (1401020792 : u32),
                                                  (989618631 : u32),
                                                  (1545325389 : u32),
                                                  (1715719711 : u32),
                                                  (755691969 : u32),
                                                  (150307788 : u32),
                                                  (1567618575 : u32),
                                                  (1663353317 : u32),
                                                  (1950429111 : u32)]),
                            (RustArray.ofVec #v[(1891637550 : u32),
                                                  (192082241 : u32),
                                                  (1080533265 : u32),
                                                  (1463323727 : u32),
                                                  (890243564 : u32),
                                                  (158646617 : u32),
                                                  (1402624179 : u32),
                                                  (59510015 : u32),
                                                  (1198261138 : u32),
                                                  (1065075039 : u32),
                                                  (1150410028 : u32),
                                                  (1293938517 : u32),
                                                  (76770019 : u32),
                                                  (1478577620 : u32),
                                                  (1748789933 : u32),
                                                  (457372011 : u32),
                                                  (1841795381 : u32),
                                                  (760115692 : u32),
                                                  (1042892522 : u32),
                                                  (1507649755 : u32),
                                                  (1827572010 : u32),
                                                  (1206940496 : u32),
                                                  (1896271507 : u32),
                                                  (1003792297 : u32)]),
                            (RustArray.ofVec #v[(738091882 : u32),
                                                  (1124078057 : u32),
                                                  (1889898 : u32),
                                                  (813674331 : u32),
                                                  (228520958 : u32),
                                                  (1832911930 : u32),
                                                  (781141772 : u32),
                                                  (459826664 : u32),
                                                  (202271745 : u32),
                                                  (1296144415 : u32),
                                                  (1111203133 : u32),
                                                  (1090783436 : u32),
                                                  (641665156 : u32),
                                                  (1393671120 : u32),
                                                  (1303271640 : u32),
                                                  (809508074 : u32),
                                                  (162506101 : u32),
                                                  (1262312258 : u32),
                                                  (1672219447 : u32),
                                                  (1608891156 : u32),
                                                  (1380248020 : u32),
                                                  (555490988 : u32),
                                                  (112090494 : u32),
                                                  (1351808603 : u32)]),
                            (RustArray.ofVec #v[(671614470 : u32),
                                                  (1987330347 : u32),
                                                  (128914032 : u32),
                                                  (1130840447 : u32),
                                                  (1355488298 : u32),
                                                  (264825773 : u32),
                                                  (672574396 : u32),
                                                  (1058448794 : u32),
                                                  (1298349090 : u32),
                                                  (1910233239 : u32),
                                                  (1865103244 : u32),
                                                  (154592857 : u32),
                                                  (195689763 : u32),
                                                  (1234605904 : u32),
                                                  (1274894892 : u32),
                                                  (530266673 : u32),
                                                  (1631576439 : u32),
                                                  (773264541 : u32),
                                                  (88285098 : u32),
                                                  (117895227 : u32),
                                                  (1518407379 : u32),
                                                  (877929693 : u32),
                                                  (823242958 : u32),
                                                  (192369619 : u32)]),
                            (RustArray.ofVec #v[(1759254541 : u32),
                                                  (1028743759 : u32),
                                                  (527617667 : u32),
                                                  (1869138177 : u32),
                                                  (972035075 : u32),
                                                  (1044999316 : u32),
                                                  (260923426 : u32),
                                                  (999354987 : u32),
                                                  (180682521 : u32),
                                                  (951079326 : u32),
                                                  (516337512 : u32),
                                                  (1177699304 : u32),
                                                  (894244668 : u32),
                                                  (331131557 : u32),
                                                  (675338309 : u32),
                                                  (869676561 : u32),
                                                  (1734188505 : u32),
                                                  (1846653664 : u32),
                                                  (1029860243 : u32),
                                                  (101912695 : u32),
                                                  (1045466369 : u32),
                                                  (1609474319 : u32),
                                                  (180316556 : u32),
                                                  (172701782 : u32)]),
                            (RustArray.ofVec #v[(307097061 : u32),
                                                  (1745215045 : u32),
                                                  (1942994748 : u32),
                                                  (1207489031 : u32),
                                                  (1200291995 : u32),
                                                  (1740512302 : u32),
                                                  (733389557 : u32),
                                                  (1660566353 : u32),
                                                  (1092211140 : u32),
                                                  (1162053316 : u32),
                                                  (1173292890 : u32),
                                                  (1493869455 : u32),
                                                  (266137465 : u32),
                                                  (92403633 : u32),
                                                  (332882885 : u32),
                                                  (102653983 : u32),
                                                  (870122976 : u32),
                                                  (603971906 : u32),
                                                  (724151438 : u32),
                                                  (1203995382 : u32),
                                                  (1048083590 : u32),
                                                  (1059106957 : u32),
                                                  (1301503482 : u32),
                                                  (977019871 : u32)]),
                            (RustArray.ofVec #v[(1302179073 : u32),
                                                  (1314774492 : u32),
                                                  (1915464449 : u32),
                                                  (441570766 : u32),
                                                  (669341367 : u32),
                                                  (1067552737 : u32),
                                                  (1829710168 : u32),
                                                  (811202513 : u32),
                                                  (587536855 : u32),
                                                  (961737002 : u32),
                                                  (1180802342 : u32),
                                                  (1655237624 : u32),
                                                  (1682944171 : u32),
                                                  (580973526 : u32),
                                                  (570259023 : u32),
                                                  (847667273 : u32),
                                                  (1083257841 : u32),
                                                  (375892130 : u32),
                                                  (111593399 : u32),
                                                  (1867716111 : u32),
                                                  (658182610 : u32),
                                                  (51866718 : u32),
                                                  (1928969210 : u32),
                                                  (1942928018 : u32)]),
                            (RustArray.ofVec #v[(1558116382 : u32),
                                                  (20525702 : u32),
                                                  (1188752903 : u32),
                                                  (106789799 : u32),
                                                  (1389833584 : u32),
                                                  (1001081700 : u32),
                                                  (1792686147 : u32),
                                                  (801504237 : u32),
                                                  (1997365681 : u32),
                                                  (1461037802 : u32),
                                                  (65998481 : u32),
                                                  (1974912881 : u32),
                                                  (606789472 : u32),
                                                  (13683277 : u32),
                                                  (918610825 : u32),
                                                  (1711450203 : u32),
                                                  (438976048 : u32),
                                                  (149438510 : u32),
                                                  (1329755157 : u32),
                                                  (1285591243 : u32),
                                                  (288088075 : u32),
                                                  (1754541169 : u32),
                                                  (1525262809 : u32),
                                                  (265815808 : u32)]),
                            (RustArray.ofVec #v[(1896429495 : u32),
                                                  (572931466 : u32),
                                                  (1765770021 : u32),
                                                  (1524042136 : u32),
                                                  (863795251 : u32),
                                                  (1693706410 : u32),
                                                  (1356741673 : u32),
                                                  (1531789568 : u32),
                                                  (1694313935 : u32),
                                                  (1994637140 : u32),
                                                  (472205185 : u32),
                                                  (1426107658 : u32),
                                                  (557777376 : u32),
                                                  (92668567 : u32),
                                                  (168822475 : u32),
                                                  (1996786329 : u32),
                                                  (1566713982 : u32),
                                                  (364497158 : u32),
                                                  (524906240 : u32),
                                                  (12634080 : u32),
                                                  (841704562 : u32),
                                                  (663708359 : u32),
                                                  (230812601 : u32),
                                                  (1305848222 : u32)]),
                            (RustArray.ofVec #v[(853915962 : u32),
                                                  (1580727053 : u32),
                                                  (1208240351 : u32),
                                                  (457653276 : u32),
                                                  (1324360850 : u32),
                                                  (30624868 : u32),
                                                  (405006464 : u32),
                                                  (358974769 : u32),
                                                  (1528376025 : u32),
                                                  (1498776016 : u32),
                                                  (324505992 : u32),
                                                  (1250803755 : u32),
                                                  (293184808 : u32),
                                                  (842187205 : u32),
                                                  (1979871109 : u32),
                                                  (600879446 : u32),
                                                  (924154107 : u32),
                                                  (1411089103 : u32),
                                                  (1893083281 : u32),
                                                  (907740982 : u32),
                                                  (1955983665 : u32),
                                                  (1298086959 : u32),
                                                  (348728829 : u32),
                                                  (146038311 : u32)]),
                            (RustArray.ofVec #v[(980751231 : u32),
                                                  (1195128645 : u32),
                                                  (1267846180 : u32),
                                                  (215427161 : u32),
                                                  (236971760 : u32),
                                                  (325938356 : u32),
                                                  (1988847481 : u32),
                                                  (213503360 : u32),
                                                  (334261497 : u32),
                                                  (583555654 : u32),
                                                  (1058048827 : u32),
                                                  (413245576 : u32),
                                                  (1046105695 : u32),
                                                  (1134466690 : u32),
                                                  (1634189787 : u32),
                                                  (66021417 : u32),
                                                  (1206665593 : u32),
                                                  (887797965 : u32),
                                                  (1903483286 : u32),
                                                  (1735422201 : u32),
                                                  (999301583 : u32),
                                                  (1602908558 : u32),
                                                  (1119705912 : u32),
                                                  (496066664 : u32)]),
                            (RustArray.ofVec #v[(1195996835 : u32),
                                                  (1963589100 : u32),
                                                  (322973716 : u32),
                                                  (1740477364 : u32),
                                                  (490289563 : u32),
                                                  (1066595225 : u32),
                                                  (1210199014 : u32),
                                                  (1628817086 : u32),
                                                  (782318021 : u32),
                                                  (1283307421 : u32),
                                                  (1528195503 : u32),
                                                  (1337337976 : u32),
                                                  (1657827562 : u32),
                                                  (1662824841 : u32),
                                                  (136210395 : u32),
                                                  (933134360 : u32),
                                                  (325488629 : u32),
                                                  (930204972 : u32),
                                                  (1661646646 : u32),
                                                  (1403419116 : u32),
                                                  (308104397 : u32),
                                                  (156475913 : u32),
                                                  (922601695 : u32),
                                                  (2008966811 : u32)]),
                            (RustArray.ofVec #v[(1531881466 : u32),
                                                  (401227607 : u32),
                                                  (1872774479 : u32),
                                                  (133107365 : u32),
                                                  (333164731 : u32),
                                                  (422745921 : u32),
                                                  (17310320 : u32),
                                                  (1040042426 : u32),
                                                  (89038513 : u32),
                                                  (1608282027 : u32),
                                                  (1757743453 : u32),
                                                  (941665211 : u32),
                                                  (415538854 : u32),
                                                  (782112906 : u32),
                                                  (719764110 : u32),
                                                  (857139636 : u32),
                                                  (295236934 : u32),
                                                  (1833479308 : u32),
                                                  (299528607 : u32),
                                                  (1300853225 : u32),
                                                  (1051566517 : u32),
                                                  (841734586 : u32),
                                                  (693288998 : u32),
                                                  (907389318 : u32)]),
                            (RustArray.ofVec #v[(474871249 : u32),
                                                  (1023570336 : u32),
                                                  (293755700 : u32),
                                                  (1140983186 : u32),
                                                  (279825535 : u32),
                                                  (1953645858 : u32),
                                                  (444415175 : u32),
                                                  (1006968146 : u32),
                                                  (1753913242 : u32),
                                                  (433318172 : u32),
                                                  (1867339844 : u32),
                                                  (993419909 : u32),
                                                  (1001103348 : u32),
                                                  (1336617011 : u32),
                                                  (1404086572 : u32),
                                                  (255173107 : u32),
                                                  (174522548 : u32),
                                                  (563930063 : u32),
                                                  (1487580563 : u32),
                                                  (22833522 : u32),
                                                  (1656698707 : u32),
                                                  (1275863723 : u32),
                                                  (102079916 : u32),
                                                  (1954183131 : u32)]),
                            (RustArray.ofVec #v[(304801400 : u32),
                                                  (67830907 : u32),
                                                  (225462907 : u32),
                                                  (1098094157 : u32),
                                                  (1574131931 : u32),
                                                  (310658596 : u32),
                                                  (434309073 : u32),
                                                  (1416420948 : u32),
                                                  (542938582 : u32),
                                                  (1867123776 : u32),
                                                  (1531134459 : u32),
                                                  (1718319318 : u32),
                                                  (71548538 : u32),
                                                  (999823361 : u32),
                                                  (1642260784 : u32),
                                                  (141982099 : u32),
                                                  (1929491374 : u32),
                                                  (864691069 : u32),
                                                  (1549940176 : u32),
                                                  (1317484777 : u32),
                                                  (985634333 : u32),
                                                  (138022525 : u32),
                                                  (838942164 : u32),
                                                  (108106978 : u32)]),
                            (RustArray.ofVec #v[(211131242 : u32),
                                                  (1666815111 : u32),
                                                  (1784530620 : u32),
                                                  (585566970 : u32),
                                                  (830018372 : u32),
                                                  (417046673 : u32),
                                                  (1336987604 : u32),
                                                  (1970003816 : u32),
                                                  (1681775946 : u32),
                                                  (389521626 : u32),
                                                  (1268898383 : u32),
                                                  (1046021040 : u32),
                                                  (420757312 : u32),
                                                  (1076039398 : u32),
                                                  (881633447 : u32),
                                                  (249620674 : u32),
                                                  (802343609 : u32),
                                                  (567820249 : u32),
                                                  (372575445 : u32),
                                                  (712308945 : u32),
                                                  (1061489096 : u32),
                                                  (1576988309 : u32),
                                                  (1615578985 : u32),
                                                  (1950891042 : u32)]),
                            (RustArray.ofVec #v[(1918433357 : u32),
                                                  (954873624 : u32),
                                                  (1561051207 : u32),
                                                  (1192206504 : u32),
                                                  (874998007 : u32),
                                                  (1853702084 : u32),
                                                  (742068765 : u32),
                                                  (1826308784 : u32),
                                                  (1367191182 : u32),
                                                  (57118950 : u32),
                                                  (905300999 : u32),
                                                  (188948710 : u32),
                                                  (1539716463 : u32),
                                                  (1038024658 : u32),
                                                  (1661421044 : u32),
                                                  (1026114709 : u32),
                                                  (207822320 : u32),
                                                  (934840306 : u32),
                                                  (327450161 : u32),
                                                  (555203968 : u32),
                                                  (833722432 : u32),
                                                  (1019879911 : u32),
                                                  (899546883 : u32),
                                                  (889947965 : u32)]),
                            (RustArray.ofVec #v[(1846126060 : u32),
                                                  (1915218698 : u32),
                                                  (750132296 : u32),
                                                  (1119439704 : u32),
                                                  (1952718685 : u32),
                                                  (726374159 : u32),
                                                  (1591529737 : u32),
                                                  (373757480 : u32),
                                                  (1320019259 : u32),
                                                  (360156120 : u32),
                                                  (258201324 : u32),
                                                  (1967259437 : u32),
                                                  (1719829466 : u32),
                                                  (1879386795 : u32),
                                                  (1684272452 : u32),
                                                  (366311406 : u32),
                                                  (1666148280 : u32),
                                                  (965070984 : u32),
                                                  (72728031 : u32),
                                                  (1009530448 : u32),
                                                  (1379832268 : u32),
                                                  (332627408 : u32),
                                                  (1841622256 : u32),
                                                  (30571725 : u32)]),
                            (RustArray.ofVec #v[(158921081 : u32),
                                                  (1784548032 : u32),
                                                  (90363221 : u32),
                                                  (691600719 : u32),
                                                  (1143770775 : u32),
                                                  (674145770 : u32),
                                                  (651382740 : u32),
                                                  (1303038925 : u32),
                                                  (1261087588 : u32),
                                                  (435350602 : u32),
                                                  (121422235 : u32),
                                                  (2004294850 : u32),
                                                  (1890436135 : u32),
                                                  (1252722819 : u32),
                                                  (1335110360 : u32),
                                                  (1745807512 : u32),
                                                  (1943888031 : u32),
                                                  (792225600 : u32),
                                                  (1514435954 : u32),
                                                  (1180826620 : u32),
                                                  (288942111 : u32),
                                                  (776716325 : u32),
                                                  (1263449775 : u32),
                                                  (1138880868 : u32)]),
                            (RustArray.ofVec #v[(264564686 : u32),
                                                  (1659681241 : u32),
                                                  (872257220 : u32),
                                                  (1771526689 : u32),
                                                  (340306609 : u32),
                                                  (539079469 : u32),
                                                  (41295736 : u32),
                                                  (919808922 : u32),
                                                  (1240628843 : u32),
                                                  (1799096581 : u32),
                                                  (1052997216 : u32),
                                                  (692820719 : u32),
                                                  (1871820490 : u32),
                                                  (1922132774 : u32),
                                                  (99656842 : u32),
                                                  (6444290 : u32),
                                                  (368947973 : u32),
                                                  (377311764 : u32),
                                                  (1182396016 : u32),
                                                  (1953338291 : u32),
                                                  (1039435657 : u32),
                                                  (1331586806 : u32),
                                                  (1460667993 : u32),
                                                  (1834032191 : u32)]),
                            (RustArray.ofVec #v[(1095002134 : u32),
                                                  (883532894 : u32),
                                                  (293115906 : u32),
                                                  (242120003 : u32),
                                                  (893626516 : u32),
                                                  (656057743 : u32),
                                                  (1813576382 : u32),
                                                  (1729552513 : u32),
                                                  (1316928927 : u32),
                                                  (1791226258 : u32),
                                                  (1535232125 : u32),
                                                  (1297716477 : u32),
                                                  (186507834 : u32),
                                                  (1470213001 : u32),
                                                  (761600655 : u32),
                                                  (485391603 : u32),
                                                  (783994008 : u32),
                                                  (258112136 : u32),
                                                  (283076917 : u32),
                                                  (242313101 : u32),
                                                  (886333754 : u32),
                                                  (1736121148 : u32),
                                                  (1199646756 : u32),
                                                  (1103537534 : u32)]),
                            (RustArray.ofVec #v[(353638884 : u32),
                                                  (1334953001 : u32),
                                                  (1593112576 : u32),
                                                  (1718547577 : u32),
                                                  (849442650 : u32),
                                                  (1913448800 : u32),
                                                  (507484918 : u32),
                                                  (1905100877 : u32),
                                                  (437636391 : u32),
                                                  (861766924 : u32),
                                                  (1237946275 : u32),
                                                  (1774979787 : u32),
                                                  (486486841 : u32),
                                                  (1409868548 : u32),
                                                  (1315866222 : u32),
                                                  (1004756277 : u32),
                                                  (1042229509 : u32),
                                                  (603052045 : u32),
                                                  (1714300831 : u32),
                                                  (339076259 : u32),
                                                  (1977801989 : u32),
                                                  (159933002 : u32),
                                                  (930749189 : u32),
                                                  (1982393493 : u32)]),
                            (RustArray.ofVec #v[(1627457601 : u32),
                                                  (1119603897 : u32),
                                                  (126680411 : u32),
                                                  (1229946045 : u32),
                                                  (1743259203 : u32),
                                                  (700962576 : u32),
                                                  (879695426 : u32),
                                                  (1582346377 : u32),
                                                  (1748671772 : u32),
                                                  (1837486239 : u32),
                                                  (1959272838 : u32),
                                                  (705285120 : u32),
                                                  (1459837508 : u32),
                                                  (1481753234 : u32),
                                                  (53211633 : u32),
                                                  (697434228 : u32),
                                                  (296949829 : u32),
                                                  (979641589 : u32),
                                                  (366122795 : u32),
                                                  (1061145335 : u32),
                                                  (65771057 : u32),
                                                  (1738923932 : u32),
                                                  (1807253439 : u32),
                                                  (1255733430 : u32)])])))
    (by rfl)

def ___1 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          24) (← (rust_primitives.unsize BABYBEAR_POSEIDON1_RC_24))))
        ==? (← ((← ((2 : usize) *? BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS))
          +? BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24))))))
    (by rfl)

--  Create a default width-16 Poseidon1 permutation for BabyBear.
@[spec]
def default_babybear_poseidon1_16 (_ : rust_primitives.hax.Tuple0) :
    RustM
    (p3_poseidon1.Poseidon1
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        p3_baby_bear.mds.MDSBabyBearData
        ((16 : usize)))
      (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((16 : usize))
        BabyBearPoseidonParameters)
      ((16 : usize))
      ((7 : u64)))
    := do
  (p3_poseidon1.Impl_1.new
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      p3_baby_bear.mds.MDSBabyBearData
      ((16 : usize)))
    (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize))
      BabyBearPoseidonParameters)
    ((16 : usize))
    ((7 : u64))
    (p3_poseidon1.Poseidon1Constants.mk
      (rounds_f := (← ((2 : usize) *? BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS)))
      (rounds_p := BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16)
      (mds_circ_col := (p3_monty_31.mds.MDSUtils.MATRIX_CIRC_MDS_16_COL
        p3_baby_bear.mds.MDSBabyBearData))
      (round_constants := (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        16) (← (rust_primitives.unsize BABYBEAR_POSEIDON1_RC_16)))))))

--  Create a default width-24 Poseidon1 permutation for BabyBear.
@[spec]
def default_babybear_poseidon1_24 (_ : rust_primitives.hax.Tuple0) :
    RustM
    (p3_poseidon1.Poseidon1
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        p3_baby_bear.mds.MDSBabyBearData
        ((24 : usize)))
      (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((24 : usize))
        BabyBearPoseidonParameters)
      ((24 : usize))
      ((7 : u64)))
    := do
  (p3_poseidon1.Impl_1.new
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon1.Poseidon1ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      p3_baby_bear.mds.MDSBabyBearData
      ((24 : usize)))
    (p3_monty_31.no_packing.poseidon1.Poseidon1InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize))
      BabyBearPoseidonParameters)
    ((24 : usize))
    ((7 : u64))
    (p3_poseidon1.Poseidon1Constants.mk
      (rounds_f := (← ((2 : usize) *? BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS)))
      (rounds_p := BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24)
      (mds_circ_col := (p3_monty_31.mds.MDSUtils.MATRIX_CIRC_MDS_24_COL
        p3_baby_bear.mds.MDSBabyBearData))
      (round_constants := (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        24) (← (rust_primitives.unsize BABYBEAR_POSEIDON1_RC_24)))))))

end p3_baby_bear.poseidon1


namespace p3_baby_bear.poseidon2

abbrev Poseidon2ExternalLayerBabyBear (WIDTH : usize) :
  Type :=
  (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    (WIDTH))

--  Number of full rounds per half for BabyBear Poseidon2 (`RF / 2`).
-- 
--  The total number of full rounds is `RF = 8` (4 beginning + 4 ending).
--  Follows the Poseidon2 paper's security analysis with a +2 RF margin.
def BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS : usize := (4 : usize)

--  Number of partial rounds for BabyBear Poseidon2 (width 16).
-- 
--  Derived from the Gröbner basis bound in the Poseidon2 paper (Eq. 1, R_GB term 3):
-- 
--    R_GB ≥ t − 7 + log_α(2) · min{κ/(t+1), log_2(p)/2}
--         = 9 + 0.3562 · min{7.53, 15.5} = 11.682
-- 
--  With the +7.5% security margin: ⌈1.075 × 11.682⌉ = 13.
def BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16 : usize := (13 : usize)

--  Number of partial rounds for BabyBear Poseidon2 (width 24).
-- 
--  Same Gröbner basis bound as width 16:
-- 
--    R_GB ≥ 17 + 0.3562 · min{5.12, 15.5} = 18.824
-- 
--  With the +7.5% security margin: ⌈1.075 × 18.824⌉ = 21.
def BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24 : usize := (21 : usize)

--  Number of partial rounds for BabyBear Poseidon2 (width 32).
-- 
--  The official round number script yields R_P = 30 for this configuration
--  (matching the Grain LFSR parameters used to generate the round constants below).
def BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_32 : usize := (30 : usize)

--  Round constants for width-16 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=16, R_F=8, R_P=13
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 16`.
-- 
--  Layout: external_initial (4 rounds × 16 elements).
def BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  16)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(1774958255 : u32),
                                                (1185780729 : u32),
                                                (1621102414 : u32),
                                                (1796380621 : u32),
                                                (588815102 : u32),
                                                (1932426223 : u32),
                                                (1925334750 : u32),
                                                (747903232 : u32),
                                                (89648862 : u32),
                                                (360728943 : u32),
                                                (977184635 : u32),
                                                (1425273457 : u32),
                                                (256487465 : u32),
                                                (1200041953 : u32),
                                                (572403254 : u32),
                                                (448208942 : u32)]),
                            (RustArray.ofVec #v[(1215789478 : u32),
                                                  (944884184 : u32),
                                                  (953948096 : u32),
                                                  (547326025 : u32),
                                                  (646827752 : u32),
                                                  (889997530 : u32),
                                                  (1536873262 : u32),
                                                  (86189867 : u32),
                                                  (1065944411 : u32),
                                                  (32019634 : u32),
                                                  (333311454 : u32),
                                                  (456061748 : u32),
                                                  (1963448500 : u32),
                                                  (1827584334 : u32),
                                                  (1391160226 : u32),
                                                  (1348741381 : u32)]),
                            (RustArray.ofVec #v[(88424255 : u32),
                                                  (104111868 : u32),
                                                  (1763866748 : u32),
                                                  (79691676 : u32),
                                                  (1988915530 : u32),
                                                  (1050669594 : u32),
                                                  (359890076 : u32),
                                                  (573163527 : u32),
                                                  (222820492 : u32),
                                                  (159256268 : u32),
                                                  (669703072 : u32),
                                                  (763177444 : u32),
                                                  (889367200 : u32),
                                                  (256335831 : u32),
                                                  (704371273 : u32),
                                                  (25886717 : u32)]),
                            (RustArray.ofVec #v[(51754520 : u32),
                                                  (1833211857 : u32),
                                                  (454499742 : u32),
                                                  (1384520381 : u32),
                                                  (777848065 : u32),
                                                  (1053320300 : u32),
                                                  (1851729162 : u32),
                                                  (344647910 : u32),
                                                  (401996362 : u32),
                                                  (1046925956 : u32),
                                                  (5351995 : u32),
                                                  (1212119315 : u32),
                                                  (754867989 : u32),
                                                  (36972490 : u32),
                                                  (751272725 : u32),
                                                  (506915399 : u32)])])))
    (by rfl)

def const_check_2 : rust_primitives.hax.Tuple0 := -- PATCHED: hax emitted `def _`
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          16)
          (← (rust_primitives.unsize
            BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-16 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=16, R_F=8, R_P=13
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 16`.
-- 
--  Layout: external_final (4 rounds × 16 elements).
def BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  16)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(1922082829 : u32),
                                                (1870549801 : u32),
                                                (1502529704 : u32),
                                                (1990744480 : u32),
                                                (1700391016 : u32),
                                                (1702593455 : u32),
                                                (321330495 : u32),
                                                (528965731 : u32),
                                                (183414327 : u32),
                                                (1886297254 : u32),
                                                (1178602734 : u32),
                                                (1923111974 : u32),
                                                (744004766 : u32),
                                                (549271463 : u32),
                                                (1781349648 : u32),
                                                (542259047 : u32)]),
                            (RustArray.ofVec #v[(1536158148 : u32),
                                                  (715456982 : u32),
                                                  (503426110 : u32),
                                                  (340311124 : u32),
                                                  (1558555932 : u32),
                                                  (1226350925 : u32),
                                                  (742828095 : u32),
                                                  (1338992758 : u32),
                                                  (1641600456 : u32),
                                                  (1843351545 : u32),
                                                  (301835475 : u32),
                                                  (43203215 : u32),
                                                  (386838401 : u32),
                                                  (1520185679 : u32),
                                                  (1235297680 : u32),
                                                  (904680097 : u32)]),
                            (RustArray.ofVec #v[(1491801617 : u32),
                                                  (1581784677 : u32),
                                                  (913384905 : u32),
                                                  (247083962 : u32),
                                                  (532844013 : u32),
                                                  (107190701 : u32),
                                                  (213827818 : u32),
                                                  (1979521776 : u32),
                                                  (1358282574 : u32),
                                                  (1681743681 : u32),
                                                  (1867507480 : u32),
                                                  (1530706910 : u32),
                                                  (507181886 : u32),
                                                  (695185447 : u32),
                                                  (1172395131 : u32),
                                                  (1250800299 : u32)]),
                            (RustArray.ofVec #v[(1503161625 : u32),
                                                  (817684387 : u32),
                                                  (498481458 : u32),
                                                  (494676004 : u32),
                                                  (1404253825 : u32),
                                                  (108246855 : u32),
                                                  (59414691 : u32),
                                                  (744214112 : u32),
                                                  (890862029 : u32),
                                                  (1342765939 : u32),
                                                  (1417398904 : u32),
                                                  (1897591937 : u32),
                                                  (1066647396 : u32),
                                                  (1682806907 : u32),
                                                  (1015795079 : u32),
                                                  (1619482808 : u32)])])))
    (by rfl)

def ___1 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          16)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-16 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=16, R_F=8, R_P=13
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 16`.
-- 
--  Layout: internal (13 scalar constants).
def BABYBEAR_POSEIDON2_RC_16_INTERNAL :
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  13)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((13 : usize))
      (RustArray.ofVec #v[(1518359488 : u32),
                            (1765533241 : u32),
                            (945325693 : u32),
                            (422793067 : u32),
                            (311365592 : u32),
                            (1311448267 : u32),
                            (1629555936 : u32),
                            (1009879353 : u32),
                            (190525218 : u32),
                            (786108885 : u32),
                            (557776863 : u32),
                            (212616710 : u32),
                            (605745517 : u32)])))
    (by rfl)

def ___2 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_16_INTERNAL))))
        ==? BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16))))
    (by rfl)

--  Round constants for width-24 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=24, R_F=8, R_P=21
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 24`.
-- 
--  Layout: external_initial (4 rounds × 24 elements).
def BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  24)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(262278199 : u32),
                                                (127253399 : u32),
                                                (314968988 : u32),
                                                (246143118 : u32),
                                                (157582794 : u32),
                                                (118043943 : u32),
                                                (454905424 : u32),
                                                (815798990 : u32),
                                                (1004040026 : u32),
                                                (1773108264 : u32),
                                                (1066694495 : u32),
                                                (1930780904 : u32),
                                                (1180307149 : u32),
                                                (1464793095 : u32),
                                                (1660766320 : u32),
                                                (1389166148 : u32),
                                                (343354132 : u32),
                                                (1307439985 : u32),
                                                (638242172 : u32),
                                                (525458520 : u32),
                                                (1964135730 : u32),
                                                (1751797115 : u32),
                                                (1421525369 : u32),
                                                (831813382 : u32)]),
                            (RustArray.ofVec #v[(695835963 : u32),
                                                  (1845603984 : u32),
                                                  (540703332 : u32),
                                                  (1333667262 : u32),
                                                  (1917861751 : u32),
                                                  (1170029417 : u32),
                                                  (1989924532 : u32),
                                                  (1518763784 : u32),
                                                  (1339793538 : u32),
                                                  (622609176 : u32),
                                                  (686842369 : u32),
                                                  (1737016378 : u32),
                                                  (1282239129 : u32),
                                                  (897025192 : u32),
                                                  (716894289 : u32),
                                                  (1997503974 : u32),
                                                  (395622276 : u32),
                                                  (1201063290 : u32),
                                                  (1917549072 : u32),
                                                  (1150912935 : u32),
                                                  (1687379185 : u32),
                                                  (1507936940 : u32),
                                                  (241306552 : u32),
                                                  (989176635 : u32)]),
                            (RustArray.ofVec #v[(1147522062 : u32),
                                                  (27129487 : u32),
                                                  (1257820264 : u32),
                                                  (142102402 : u32),
                                                  (217046702 : u32),
                                                  (1664590951 : u32),
                                                  (855276054 : u32),
                                                  (1215259350 : u32),
                                                  (946500736 : u32),
                                                  (552696906 : u32),
                                                  (1424297384 : u32),
                                                  (538103555 : u32),
                                                  (1608853840 : u32),
                                                  (162510541 : u32),
                                                  (623051854 : u32),
                                                  (1549062383 : u32),
                                                  (1908416316 : u32),
                                                  (1622328571 : u32),
                                                  (1079030649 : u32),
                                                  (1584033957 : u32),
                                                  (1099252725 : u32),
                                                  (1910423126 : u32),
                                                  (447555988 : u32),
                                                  (862495875 : u32)]),
                            (RustArray.ofVec #v[(128479034 : u32),
                                                  (1587822577 : u32),
                                                  (608401422 : u32),
                                                  (1290028279 : u32),
                                                  (342857858 : u32),
                                                  (825405577 : u32),
                                                  (427731030 : u32),
                                                  (1718628547 : u32),
                                                  (588764636 : u32),
                                                  (204228775 : u32),
                                                  (1454563174 : u32),
                                                  (1740472809 : u32),
                                                  (1338899225 : u32),
                                                  (1269493554 : u32),
                                                  (53007114 : u32),
                                                  (1647670797 : u32),
                                                  (306391314 : u32),
                                                  (172614232 : u32),
                                                  (51256176 : u32),
                                                  (1221257987 : u32),
                                                  (1239734761 : u32),
                                                  (273790406 : u32),
                                                  (1781980094 : u32),
                                                  (1291790245 : u32)])])))
    (by rfl)

def ___3 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          24)
          (← (rust_primitives.unsize
            BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-24 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=24, R_F=8, R_P=21
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 24`.
-- 
--  Layout: external_final (4 rounds × 24 elements).
def BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  24)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(53041581 : u32),
                                                (723038058 : u32),
                                                (1439947916 : u32),
                                                (1136469704 : u32),
                                                (205609311 : u32),
                                                (1883820770 : u32),
                                                (14387587 : u32),
                                                (720724951 : u32),
                                                (1854174607 : u32),
                                                (1629316321 : u32),
                                                (530151394 : u32),
                                                (1679178250 : u32),
                                                (1549779579 : u32),
                                                (48375137 : u32),
                                                (976057819 : u32),
                                                (463976218 : u32),
                                                (875839332 : u32),
                                                (1946596189 : u32),
                                                (434078361 : u32),
                                                (1878280202 : u32),
                                                (1363837384 : u32),
                                                (1470845646 : u32),
                                                (1792450386 : u32),
                                                (1040977421 : u32)]),
                            (RustArray.ofVec #v[(1209164052 : u32),
                                                  (714957516 : u32),
                                                  (390340387 : u32),
                                                  (1213686459 : u32),
                                                  (790726260 : u32),
                                                  (117294666 : u32),
                                                  (140621810 : u32),
                                                  (993455846 : u32),
                                                  (1889603648 : u32),
                                                  (78845751 : u32),
                                                  (925018226 : u32),
                                                  (708123747 : u32),
                                                  (1647665372 : u32),
                                                  (1649953458 : u32),
                                                  (942439428 : u32),
                                                  (1006235079 : u32),
                                                  (238616145 : u32),
                                                  (930036496 : u32),
                                                  (1401020792 : u32),
                                                  (989618631 : u32),
                                                  (1545325389 : u32),
                                                  (1715719711 : u32),
                                                  (755691969 : u32),
                                                  (150307788 : u32)]),
                            (RustArray.ofVec #v[(1567618575 : u32),
                                                  (1663353317 : u32),
                                                  (1950429111 : u32),
                                                  (1891637550 : u32),
                                                  (192082241 : u32),
                                                  (1080533265 : u32),
                                                  (1463323727 : u32),
                                                  (890243564 : u32),
                                                  (158646617 : u32),
                                                  (1402624179 : u32),
                                                  (59510015 : u32),
                                                  (1198261138 : u32),
                                                  (1065075039 : u32),
                                                  (1150410028 : u32),
                                                  (1293938517 : u32),
                                                  (76770019 : u32),
                                                  (1478577620 : u32),
                                                  (1748789933 : u32),
                                                  (457372011 : u32),
                                                  (1841795381 : u32),
                                                  (760115692 : u32),
                                                  (1042892522 : u32),
                                                  (1507649755 : u32),
                                                  (1827572010 : u32)]),
                            (RustArray.ofVec #v[(1206940496 : u32),
                                                  (1896271507 : u32),
                                                  (1003792297 : u32),
                                                  (738091882 : u32),
                                                  (1124078057 : u32),
                                                  (1889898 : u32),
                                                  (813674331 : u32),
                                                  (228520958 : u32),
                                                  (1832911930 : u32),
                                                  (781141772 : u32),
                                                  (459826664 : u32),
                                                  (202271745 : u32),
                                                  (1296144415 : u32),
                                                  (1111203133 : u32),
                                                  (1090783436 : u32),
                                                  (641665156 : u32),
                                                  (1393671120 : u32),
                                                  (1303271640 : u32),
                                                  (809508074 : u32),
                                                  (162506101 : u32),
                                                  (1262312258 : u32),
                                                  (1672219447 : u32),
                                                  (1608891156 : u32),
                                                  (1380248020 : u32)])])))
    (by rfl)

def ___4 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          24)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-24 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=24, R_F=8, R_P=21
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 24`.
-- 
--  Layout: internal (21 scalar constants).
def BABYBEAR_POSEIDON2_RC_24_INTERNAL :
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  21)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((21 : usize))
      (RustArray.ofVec #v[(497520322 : u32),
                            (1930103076 : u32),
                            (1052077299 : u32),
                            (1540960371 : u32),
                            (924863639 : u32),
                            (1365519753 : u32),
                            (1726563304 : u32),
                            (440300254 : u32),
                            (1891545577 : u32),
                            (822033215 : u32),
                            (1111544260 : u32),
                            (308575117 : u32),
                            (1708681573 : u32),
                            (1240419708 : u32),
                            (1199068823 : u32),
                            (1186174623 : u32),
                            (1551596046 : u32),
                            (1886977120 : u32),
                            (1327682690 : u32),
                            (1210751726 : u32),
                            (1810596765 : u32)])))
    (by rfl)

def ___5 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_24_INTERNAL))))
        ==? BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24))))
    (by rfl)

--  Round constants for width-32 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=32, R_F=8, R_P=30
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 32`.
-- 
--  Layout: external_initial (4 rounds × 32 elements).
def BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  32)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((32 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(1729160065 : u32),
                                                (27999661 : u32),
                                                (1237173535 : u32),
                                                (1103727717 : u32),
                                                (596139402 : u32),
                                                (619347324 : u32),
                                                (1801845869 : u32),
                                                (1627808090 : u32),
                                                (139024371 : u32),
                                                (360461876 : u32),
                                                (1303224591 : u32),
                                                (1826595949 : u32),
                                                (468846782 : u32),
                                                (1540501420 : u32),
                                                (1284230111 : u32),
                                                (670536848 : u32),
                                                (1883853842 : u32),
                                                (708936782 : u32),
                                                (1371236849 : u32),
                                                (840293409 : u32),
                                                (1729185817 : u32),
                                                (665479689 : u32),
                                                (1598897325 : u32),
                                                (1607911204 : u32),
                                                (457969805 : u32),
                                                (1469698125 : u32),
                                                (1452024111 : u32),
                                                (1745363419 : u32),
                                                (1644640041 : u32),
                                                (580839296 : u32),
                                                (995071171 : u32),
                                                (396602452 : u32)]),
                            (RustArray.ofVec #v[(1815642745 : u32),
                                                  (1215757909 : u32),
                                                  (829485578 : u32),
                                                  (862993998 : u32),
                                                  (60295857 : u32),
                                                  (19902714 : u32),
                                                  (341764315 : u32),
                                                  (1233214256 : u32),
                                                  (1185777564 : u32),
                                                  (1388073021 : u32),
                                                  (1483026647 : u32),
                                                  (1547106789 : u32),
                                                  (42886403 : u32),
                                                  (137429864 : u32),
                                                  (1968465478 : u32),
                                                  (1931810545 : u32),
                                                  (860372570 : u32),
                                                  (77628460 : u32),
                                                  (439432665 : u32),
                                                  (1400581809 : u32),
                                                  (1538215799 : u32),
                                                  (1266208109 : u32),
                                                  (1525492810 : u32),
                                                  (1724421089 : u32),
                                                  (1012175782 : u32),
                                                  (1187392508 : u32),
                                                  (1447975194 : u32),
                                                  (1390335911 : u32),
                                                  (13697837 : u32),
                                                  (724621313 : u32),
                                                  (270380023 : u32),
                                                  (788210125 : u32)]),
                            (RustArray.ofVec #v[(1245060153 : u32),
                                                  (1240462706 : u32),
                                                  (773049076 : u32),
                                                  (24012697 : u32),
                                                  (459070276 : u32),
                                                  (399125958 : u32),
                                                  (240127712 : u32),
                                                  (21088923 : u32),
                                                  (664478582 : u32),
                                                  (1715979275 : u32),
                                                  (231403068 : u32),
                                                  (384302329 : u32),
                                                  (1508300380 : u32),
                                                  (515322057 : u32),
                                                  (677410888 : u32),
                                                  (1165131676 : u32),
                                                  (1314144251 : u32),
                                                  (65938768 : u32),
                                                  (17618251 : u32),
                                                  (47991670 : u32),
                                                  (725260860 : u32),
                                                  (783004377 : u32),
                                                  (110265683 : u32),
                                                  (757314549 : u32),
                                                  (1839407793 : u32),
                                                  (354937863 : u32),
                                                  (239405946 : u32),
                                                  (1402344978 : u32),
                                                  (552692496 : u32),
                                                  (384830560 : u32),
                                                  (1278858511 : u32),
                                                  (1104518518 : u32)]),
                            (RustArray.ofVec #v[(1381234722 : u32),
                                                  (995349377 : u32),
                                                  (461326090 : u32),
                                                  (1581091164 : u32),
                                                  (1382383909 : u32),
                                                  (18976979 : u32),
                                                  (1700150144 : u32),
                                                  (487565895 : u32),
                                                  (896906185 : u32),
                                                  (1275960608 : u32),
                                                  (416280735 : u32),
                                                  (1688722012 : u32),
                                                  (789842725 : u32),
                                                  (329227825 : u32),
                                                  (1726180309 : u32),
                                                  (1682098193 : u32),
                                                  (1485282254 : u32),
                                                  (1607239622 : u32),
                                                  (1745333772 : u32),
                                                  (1091683927 : u32),
                                                  (1664911657 : u32),
                                                  (1955998065 : u32),
                                                  (1069649592 : u32),
                                                  (1164767880 : u32),
                                                  (838256850 : u32),
                                                  (1815753128 : u32),
                                                  (1837293392 : u32),
                                                  (1027811219 : u32),
                                                  (1100325527 : u32),
                                                  (1808702357 : u32),
                                                  (1015163632 : u32),
                                                  (1982457267 : u32)])])))
    (by rfl)

def ___6 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          32)
          (← (rust_primitives.unsize
            BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-32 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=32, R_F=8, R_P=30
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 32`.
-- 
--  Layout: external_final (4 rounds × 32 elements).
def BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL :
  (RustArray
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  32)
  4)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_2d_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((32 : usize))
      ((4 : usize))
      (RustArray.ofVec #v[(RustArray.ofVec #v[(2009876055 : u32),
                                                (956324903 : u32),
                                                (1221427337 : u32),
                                                (645725371 : u32),
                                                (474073733 : u32),
                                                (1102287935 : u32),
                                                (552039034 : u32),
                                                (1689090291 : u32),
                                                (333267987 : u32),
                                                (272819164 : u32),
                                                (1957554988 : u32),
                                                (768463815 : u32),
                                                (100691900 : u32),
                                                (900073961 : u32),
                                                (229582652 : u32),
                                                (978282831 : u32),
                                                (1509756957 : u32),
                                                (668078379 : u32),
                                                (423081631 : u32),
                                                (780697788 : u32),
                                                (1086399529 : u32),
                                                (108619174 : u32),
                                                (1856090989 : u32),
                                                (1845577582 : u32),
                                                (701329802 : u32),
                                                (1409730195 : u32),
                                                (1373515418 : u32),
                                                (1843148938 : u32),
                                                (1400255603 : u32),
                                                (396516543 : u32),
                                                (522814157 : u32),
                                                (432691965 : u32)]),
                            (RustArray.ofVec #v[(762330259 : u32),
                                                  (203964644 : u32),
                                                  (752517837 : u32),
                                                  (1551847904 : u32),
                                                  (498436102 : u32),
                                                  (1018571729 : u32),
                                                  (725960219 : u32),
                                                  (1644477008 : u32),
                                                  (1750593724 : u32),
                                                  (1257415528 : u32),
                                                  (851527412 : u32),
                                                  (449830291 : u32),
                                                  (767792496 : u32),
                                                  (230550350 : u32),
                                                  (1354692685 : u32),
                                                  (1870897708 : u32),
                                                  (1129068353 : u32),
                                                  (543220064 : u32),
                                                  (1760923403 : u32),
                                                  (623489725 : u32),
                                                  (288114536 : u32),
                                                  (1883282658 : u32),
                                                  (458704398 : u32),
                                                  (568099212 : u32),
                                                  (1600869168 : u32),
                                                  (1145016250 : u32),
                                                  (555389150 : u32),
                                                  (1495091760 : u32),
                                                  (1532640885 : u32),
                                                  (1666986810 : u32),
                                                  (519924336 : u32),
                                                  (1563771999 : u32)]),
                            (RustArray.ofVec #v[(1446498082 : u32),
                                                  (1855695025 : u32),
                                                  (280317484 : u32),
                                                  (69803704 : u32),
                                                  (760128835 : u32),
                                                  (1913281633 : u32),
                                                  (442196771 : u32),
                                                  (1759606503 : u32),
                                                  (638883701 : u32),
                                                  (1492323307 : u32),
                                                  (1157890731 : u32),
                                                  (613581397 : u32),
                                                  (851810645 : u32),
                                                  (727413345 : u32),
                                                  (1652127647 : u32),
                                                  (1503357582 : u32),
                                                  (1892763091 : u32),
                                                  (931579089 : u32),
                                                  (288765024 : u32),
                                                  (587145601 : u32),
                                                  (1292303440 : u32),
                                                  (665363423 : u32),
                                                  (1685977476 : u32),
                                                  (980716739 : u32),
                                                  (1252597410 : u32),
                                                  (569037473 : u32),
                                                  (1355636109 : u32),
                                                  (114753478 : u32),
                                                  (1539641623 : u32),
                                                  (1173899452 : u32),
                                                  (314824177 : u32),
                                                  (1216319589 : u32)]),
                            (RustArray.ofVec #v[(370615356 : u32),
                                                  (1552381156 : u32),
                                                  (1992086508 : u32),
                                                  (882674251 : u32),
                                                  (643656246 : u32),
                                                  (972842286 : u32),
                                                  (610808232 : u32),
                                                  (1216370940 : u32),
                                                  (173831638 : u32),
                                                  (1980583661 : u32),
                                                  (757684141 : u32),
                                                  (1636349694 : u32),
                                                  (545572213 : u32),
                                                  (1741029216 : u32),
                                                  (1030588827 : u32),
                                                  (660162122 : u32),
                                                  (507714002 : u32),
                                                  (167704630 : u32),
                                                  (810967268 : u32),
                                                  (607642520 : u32),
                                                  (1770949549 : u32),
                                                  (1369555082 : u32),
                                                  (1869300458 : u32),
                                                  (1316643372 : u32),
                                                  (861517941 : u32),
                                                  (1875492637 : u32),
                                                  (882822549 : u32),
                                                  (1076559853 : u32),
                                                  (243081034 : u32),
                                                  (1969815445 : u32),
                                                  (1831269979 : u32),
                                                  (1383829493 : u32)])])))
    (by rfl)

def ___7 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (RustArray
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          32)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL))))
        ==? BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS))))
    (by rfl)

--  Round constants for width-32 Poseidon2 on BabyBear.
-- 
--  Generated by the Grain LFSR with parameters:
--      field_type=1, alpha=7 (exp_flag=0), n=31, t=32, R_F=8, R_P=30
-- 
--  Generated by `poseidon2/generate_constants.py --field babybear --width 32`.
-- 
--  Layout: internal (30 scalar constants).
def BABYBEAR_POSEIDON2_RC_32_INTERNAL :
  (RustArray
  (p3_monty_31.monty_31.MontyField31 p3_baby_bear.baby_bear.BabyBearParameters)
  30)
  :=
  RustM.of_isOk
    (do
    (p3_monty_31.monty_31.Impl.new_array
      p3_baby_bear.baby_bear.BabyBearParameters
      ((30 : usize))
      (RustArray.ofVec #v[(1037946958 : u32),
                            (1513948504 : u32),
                            (1983802485 : u32),
                            (1735335304 : u32),
                            (2008556117 : u32),
                            (635406929 : u32),
                            (1769749264 : u32),
                            (63925983 : u32),
                            (313872248 : u32),
                            (1714672948 : u32),
                            (1114619063 : u32),
                            (1430676956 : u32),
                            (1451867570 : u32),
                            (317147102 : u32),
                            (1623361349 : u32),
                            (298470309 : u32),
                            (1057985923 : u32),
                            (1455937939 : u32),
                            (553027423 : u32),
                            (1776654280 : u32),
                            (1016141171 : u32),
                            (700970754 : u32),
                            (442032405 : u32),
                            (688992664 : u32),
                            (2003511455 : u32),
                            (512896738 : u32),
                            (560331760 : u32),
                            (684502904 : u32),
                            (428025981 : u32),
                            (1227563744 : u32)])))
    (by rfl)

def ___8 : rust_primitives.hax.Tuple0 :=
  RustM.of_isOk
    (do
    (hax_lib.assert
      (← ((← (core_models.slice.Impl.len
          (p3_monty_31.monty_31.MontyField31
            p3_baby_bear.baby_bear.BabyBearParameters)
          (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_32_INTERNAL))))
        ==? BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_32))))
    (by rfl)

--  Contains data needed to define the internal layers of the Poseidon2 permutation.
structure BabyBearInternalLayerParameters where
  -- no fields

abbrev Poseidon2InternalLayerBabyBear (WIDTH : usize) :
  Type :=
  (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    (WIDTH)
    BabyBearInternalLayerParameters)

--  An implementation of the Poseidon2 hash function specialised to run on the current architecture.
-- 
--  It acts on arrays of the form either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]`. For speed purposes,
--  wherever possible, input arrays should of the form `[BabyBear::Packing; WIDTH]`.
abbrev Poseidon2BabyBear (WIDTH : usize) :
  Type :=
  (p3_poseidon2.Poseidon2
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      (WIDTH))
    (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      (WIDTH)
      BabyBearInternalLayerParameters)
    (WIDTH)
    ((7 : u64)))

--  An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
-- 
--  This can act on `[A; WIDTH]` for any ring implementing `Algebra<BabyBear>`.
--  If you have either `[BabyBear::Packing; WIDTH]` or `[BabyBear; WIDTH]` it will be much faster
--  to use `Poseidon2BabyBear<WIDTH>` instead of building a Poseidon2 permutation using this.
abbrev GenericPoseidon2LinearLayersBabyBear :
  Type :=
  (p3_monty_31.poseidon2.GenericPoseidon2LinearLayersMonty31
    p3_baby_bear.baby_bear.BabyBearParameters
    BabyBearInternalLayerParameters)

@[instance] opaque Impl_6.AssociatedTypes :
  core_models.fmt.Debug.AssociatedTypes BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_6 :
  core_models.fmt.Debug BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_7.AssociatedTypes :
  core_models.clone.Clone.AssociatedTypes BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_7 :
  core_models.clone.Clone BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8.AssociatedTypes :
  core_models.default.Default.AssociatedTypes BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

@[instance] opaque Impl_8 :
  core_models.default.Default BabyBearInternalLayerParameters :=
  by constructor <;> exact Inhabited.default

--  Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
--  We ignore `state[0]` as it is handled separately.
@[spec]
def Impl.internal_layer_mat_mul_hoisted
    (R : Type)
    [trait_constr_internal_layer_mat_mul_hoisted_associated_type_i0 :
      p3_field.field.PrimeCharacteristicRing.AssociatedTypes
      R]
    [trait_constr_internal_layer_mat_mul_hoisted_i0 :
      p3_field.field.PrimeCharacteristicRing
      R
      ]
    (state : (RustArray R 16))
    (sum : R) :
    RustM (RustArray R 16) := do
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (1 : usize)
      (← (core_models.ops.arith.AddAssign.add_assign
        R
        R (← state[(1 : usize)]_?) (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (2 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.double
          R (← state[(2 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (3 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(3 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (4 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.dup.Dup.dup R sum))
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(4 : usize)]_?)))))
        (← (p3_field.dup.Dup.dup R (← state[(4 : usize)]_?))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (5 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(5 : usize)]_?))))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (6 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(6 : usize)]_?))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (7 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(7 : usize)]_?)))
          (← (p3_field.dup.Dup.dup R (← state[(7 : usize)]_?))))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (8 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(8 : usize)]_?))))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (9 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(9 : usize)]_?) (8 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (10 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(10 : usize)]_?) (2 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (11 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(11 : usize)]_?) (3 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (12 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(12 : usize)]_?) (27 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (13 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(13 : usize)]_?) (8 : u64))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (14 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(14 : usize)]_?) (4 : u64))))));
  let state : (RustArray R 16) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (15 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        sum
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(15 : usize)]_?) (27 : u64))))));
  (pure state)

@[reducible] instance Impl.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerBaseParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

instance Impl :
  p3_monty_31.poseidon2.InternalLayerBaseParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where
  internal_layer_mat_mul :=
    fun
      
      (R : Type)
      [trait_constr__associated_type_i0 :
        p3_field.field.PrimeCharacteristicRing.AssociatedTypes
        R]
      [trait_constr__i0 : p3_field.field.PrimeCharacteristicRing R ]
      =>
    (Impl.internal_layer_mat_mul_hoisted R)

--  Create a default width-16 Poseidon2 permutation for BabyBear.
@[spec]
def default_babybear_poseidon2_16 (_ : rust_primitives.hax.Tuple0) :
    RustM
    (p3_poseidon2.Poseidon2
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((16 : usize)))
      (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((16 : usize))
        BabyBearInternalLayerParameters)
      ((16 : usize))
      ((7 : u64)))
    := do
  (p3_poseidon2.Impl.new
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize)))
    (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((16 : usize))
      BabyBearInternalLayerParameters)
    ((16 : usize))
    ((7 : u64))
    (← (p3_poseidon2.external.Impl_4.new
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      ((16 : usize))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        16)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL))))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        16)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL))))))
    (← (alloc.slice.Impl.to_vec
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_16_INTERNAL)))))

--  Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
--  We ignore `state[0]` as it is handled separately.
@[spec]
def Impl_1.internal_layer_mat_mul_hoisted
    (R : Type)
    [trait_constr_internal_layer_mat_mul_hoisted_associated_type_i0 :
      p3_field.field.PrimeCharacteristicRing.AssociatedTypes
      R]
    [trait_constr_internal_layer_mat_mul_hoisted_i0 :
      p3_field.field.PrimeCharacteristicRing
      R
      ]
    (state : (RustArray R 24))
    (sum : R) :
    RustM (RustArray R 24) := do
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (1 : usize)
      (← (core_models.ops.arith.AddAssign.add_assign
        R
        R (← state[(1 : usize)]_?) (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (2 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.double
          R (← state[(2 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (3 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(3 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (4 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.dup.Dup.dup R sum))
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(4 : usize)]_?)))))
        (← (p3_field.dup.Dup.dup R (← state[(4 : usize)]_?))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (5 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(5 : usize)]_?))))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (6 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(6 : usize)]_?))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (7 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(7 : usize)]_?)))
          (← (p3_field.dup.Dup.dup R (← state[(7 : usize)]_?))))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (8 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(8 : usize)]_?))))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (9 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(9 : usize)]_?) (8 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (10 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(10 : usize)]_?) (2 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (11 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(11 : usize)]_?) (3 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (12 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(12 : usize)]_?) (4 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (13 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(13 : usize)]_?) (7 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (14 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(14 : usize)]_?) (9 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (15 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(15 : usize)]_?) (27 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (16 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(16 : usize)]_?) (8 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (17 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(17 : usize)]_?) (2 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (18 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(18 : usize)]_?) (3 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (19 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(19 : usize)]_?) (4 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (20 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(20 : usize)]_?) (5 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (21 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(21 : usize)]_?) (6 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (22 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(22 : usize)]_?) (7 : u64))))));
  let state : (RustArray R 24) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (23 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        sum
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(23 : usize)]_?) (27 : u64))))));
  (pure state)

@[reducible] instance Impl_1.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerBaseParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

instance Impl_1 :
  p3_monty_31.poseidon2.InternalLayerBaseParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where
  internal_layer_mat_mul :=
    fun
      
      (R : Type)
      [trait_constr__associated_type_i0 :
        p3_field.field.PrimeCharacteristicRing.AssociatedTypes
        R]
      [trait_constr__i0 : p3_field.field.PrimeCharacteristicRing R ]
      =>
    (Impl_1.internal_layer_mat_mul_hoisted R)

--  Create a default width-24 Poseidon2 permutation for BabyBear.
@[spec]
def default_babybear_poseidon2_24 (_ : rust_primitives.hax.Tuple0) :
    RustM
    (p3_poseidon2.Poseidon2
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((24 : usize)))
      (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((24 : usize))
        BabyBearInternalLayerParameters)
      ((24 : usize))
      ((7 : u64)))
    := do
  (p3_poseidon2.Impl.new
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize)))
    (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((24 : usize))
      BabyBearInternalLayerParameters)
    ((24 : usize))
    ((7 : u64))
    (← (p3_poseidon2.external.Impl_4.new
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      ((24 : usize))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        24)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL))))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        24)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL))))))
    (← (alloc.slice.Impl.to_vec
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_24_INTERNAL)))))

--  Perform the internal matrix multiplication: s -> (1 + Diag(V))s.
--  We ignore `state[0]` as it is handled separately.
@[spec]
def Impl_2.internal_layer_mat_mul_hoisted
    (R : Type)
    [trait_constr_internal_layer_mat_mul_hoisted_associated_type_i0 :
      p3_field.field.PrimeCharacteristicRing.AssociatedTypes
      R]
    [trait_constr_internal_layer_mat_mul_hoisted_i0 :
      p3_field.field.PrimeCharacteristicRing
      R
      ]
    (state : (RustArray R 32))
    (sum : R) :
    RustM (RustArray R 32) := do
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (1 : usize)
      (← (core_models.ops.arith.AddAssign.add_assign
        R
        R (← state[(1 : usize)]_?) (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (2 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.double
          R (← state[(2 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (3 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(3 : usize)]_?)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (4 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.dup.Dup.dup R sum))
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(4 : usize)]_?)))))
        (← (p3_field.dup.Dup.dup R (← state[(4 : usize)]_?))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (5 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(5 : usize)]_?))))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (6 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.halve
          R (← state[(6 : usize)]_?))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (7 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (core_models.ops.arith.Add.add
          R
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(7 : usize)]_?)))
          (← (p3_field.dup.Dup.dup R (← state[(7 : usize)]_?))))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (8 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.double
          R
          (← (p3_field.field.PrimeCharacteristicRing.double
            R (← state[(8 : usize)]_?))))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (9 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(9 : usize)]_?) (8 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (10 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(10 : usize)]_?) (2 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (11 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(11 : usize)]_?) (3 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (12 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(12 : usize)]_?) (4 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (13 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(13 : usize)]_?) (5 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (14 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(14 : usize)]_?) (6 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (15 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(15 : usize)]_?) (7 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (16 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(16 : usize)]_?) (9 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (17 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(17 : usize)]_?) (10 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (18 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(18 : usize)]_?) (12 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (19 : usize)
      (← (core_models.ops.arith.Add.add
        R
        R
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(19 : usize)]_?) (27 : u64)))
        (← (p3_field.dup.Dup.dup R sum)))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (20 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(20 : usize)]_?) (8 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (21 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(21 : usize)]_?) (2 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (22 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(22 : usize)]_?) (3 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (23 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(23 : usize)]_?) (4 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (24 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(24 : usize)]_?) (5 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (25 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(25 : usize)]_?) (6 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (26 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(26 : usize)]_?) (7 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (27 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(27 : usize)]_?) (9 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (28 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(28 : usize)]_?) (10 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (29 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(29 : usize)]_?) (12 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (30 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        (← (p3_field.dup.Dup.dup R sum))
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(30 : usize)]_?) (14 : u64))))));
  let state : (RustArray R 32) ←
    (rust_primitives.hax.monomorphized_update_at.update_at_usize
      state
      (31 : usize)
      (← (core_models.ops.arith.Sub.sub
        R
        R
        sum
        (← (p3_field.field.PrimeCharacteristicRing.div_2exp_u64
          R (← state[(31 : usize)]_?) (27 : u64))))));
  (pure state)

@[reducible] instance Impl_2.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerBaseParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((32 : usize))
  where

instance Impl_2 :
  p3_monty_31.poseidon2.InternalLayerBaseParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((32 : usize))
  where
  internal_layer_mat_mul :=
    fun
      
      (R : Type)
      [trait_constr__associated_type_i0 :
        p3_field.field.PrimeCharacteristicRing.AssociatedTypes
        R]
      [trait_constr__i0 : p3_field.field.PrimeCharacteristicRing R ]
      =>
    (p3_baby_bear.poseidon2.Impl_2.internal_layer_mat_mul_hoisted R) -- PATCHED
    -- (Impl_2.internal_layer_mat_mul_hoisted R)

--  Create a default width-32 Poseidon2 permutation for BabyBear.
@[spec]
def default_babybear_poseidon2_32 (_ : rust_primitives.hax.Tuple0) :
    RustM
    (p3_poseidon2.Poseidon2
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((32 : usize)))
      (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
        p3_baby_bear.baby_bear.BabyBearParameters
        ((32 : usize))
        BabyBearInternalLayerParameters)
      ((32 : usize))
      ((7 : u64)))
    := do
  (p3_poseidon2.Impl.new
    (p3_monty_31.monty_31.MontyField31
      p3_baby_bear.baby_bear.BabyBearParameters)
    (p3_monty_31.no_packing.poseidon2.Poseidon2ExternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((32 : usize)))
    (p3_monty_31.no_packing.poseidon2.Poseidon2InternalLayerMonty31
      p3_baby_bear.baby_bear.BabyBearParameters
      ((32 : usize))
      BabyBearInternalLayerParameters)
    ((32 : usize))
    ((7 : u64))
    (← (p3_poseidon2.external.Impl_4.new
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      ((32 : usize))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        32)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL))))
      (← (alloc.slice.Impl.to_vec
        (RustArray
        (p3_monty_31.monty_31.MontyField31
          p3_baby_bear.baby_bear.BabyBearParameters)
        32)
        (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL))))))
    (← (alloc.slice.Impl.to_vec
      (p3_monty_31.monty_31.MontyField31
        p3_baby_bear.baby_bear.BabyBearParameters)
      (← (rust_primitives.unsize BABYBEAR_POSEIDON2_RC_32_INTERNAL)))))

@[reducible] instance Impl_3.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

instance Impl_3 :
  p3_monty_31.poseidon2.InternalLayerParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((16 : usize))
  where

@[reducible] instance Impl_4.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

instance Impl_4 :
  p3_monty_31.poseidon2.InternalLayerParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((24 : usize))
  where

@[reducible] instance Impl_5.AssociatedTypes :
  p3_monty_31.poseidon2.InternalLayerParameters.AssociatedTypes
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((32 : usize))
  where

instance Impl_5 :
  p3_monty_31.poseidon2.InternalLayerParameters
  BabyBearInternalLayerParameters
  p3_baby_bear.baby_bear.BabyBearParameters
  ((32 : usize))
  where

end p3_baby_bear.poseidon2

