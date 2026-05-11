use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::any::Any;
use core::mem::transmute;

use hashbrown::HashMap;
#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::{
    GoldilocksD2Width8, NonPrimitivePreprocessedMap, NpoTypeId, Poseidon2CircuitRow,
    Poseidon2Config, Poseidon2Params, Poseidon2Trace,
};
use p3_circuit::tables::Traces;
use p3_circuit::{CircuitError, PreprocessedColumns};
use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, QuinticTrinomialExtensionField,
};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField,
    PrimeField64,
};
use p3_goldilocks::{GenericPoseidon2LinearLayersGoldilocks, Goldilocks};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_lookup::LookupAir;
use p3_lookup::folder::{ProverConstraintFolderWithLookups, VerifierConstraintFolderWithLookups};
use p3_lookup::lookup_traits::Lookup;
use p3_matrix::dense::RowMajorMatrix;
use p3_poseidon2_circuit_air::*;
use p3_uni_stark::{
    ProverConstraintFolder, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
    VerifierConstraintFolder,
};
use p3_util::log2_ceil_usize;

use super::dynamic_air::{BatchAir, BatchTableInstance, DynamicAirEntry, TableProver};
use crate::batch_stark_prover::{
    BABY_BEAR_MODULUS, KOALA_BEAR_MODULUS, NonPrimitiveTableEntry, TablePacking,
};
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::{BabyBearConfig, GoldilocksConfig, KoalaBearConfig, StarkField};
use crate::constraint_profile::ConstraintProfile;

/// Returns the witness-bus dimension for a D=1 Poseidon2 config given the circuit's extension
/// degree, or `None` if the scale is not supported.
///
/// Currently supported: 1 (base-field circuit) and 5 (KoalaBear quintic).
#[inline]
const fn poseidon_d1_witness_bus_dim(witness_ctl_scale: u32) -> Option<u32> {
    match witness_ctl_scale {
        1 => Some(1),
        5 => Some(5),
        _ => None,
    }
}

pub enum Poseidon2AirWrapperInner {
    BabyBearD1Width16Bus1(Box<Poseidon2CircuitAirBabyBearD1Width16>),
    BabyBearD1Width16Bus5(Box<Poseidon2CircuitAirBabyBearD1Width16WitnessBus5>),
    BabyBearD4Width16(Box<Poseidon2CircuitAirBabyBearD4Width16>),
    BabyBearD4Width24(Box<Poseidon2CircuitAirBabyBearD4Width24>),
    KoalaBearD1Width16Bus1(Box<Poseidon2CircuitAirKoalaBearD1Width16>),
    KoalaBearD1Width16Bus5(Box<Poseidon2CircuitAirKoalaBearD1Width16WitnessBus5>),
    KoalaBearD4Width16(Box<Poseidon2CircuitAirKoalaBearD4Width16>),
    KoalaBearD4Width24(Box<Poseidon2CircuitAirKoalaBearD4Width24>),
    GoldilocksD2Width8(Box<Poseidon2CircuitAirGoldilocksD2Width8>),
}

impl Poseidon2AirWrapperInner {
    pub fn width(&self) -> usize {
        match self {
            Self::BabyBearD1Width16Bus1(air) => air.width(),
            Self::BabyBearD1Width16Bus5(air) => air.width(),
            Self::BabyBearD4Width16(air) => air.width(),
            Self::BabyBearD4Width24(air) => air.width(),
            Self::KoalaBearD1Width16Bus1(air) => air.width(),
            Self::KoalaBearD1Width16Bus5(air) => air.width(),
            Self::KoalaBearD4Width16(air) => air.width(),
            Self::KoalaBearD4Width24(air) => air.width(),
            Self::GoldilocksD2Width8(air) => air.width(),
        }
    }
}

impl Clone for Poseidon2AirWrapperInner {
    fn clone(&self) -> Self {
        match self {
            Self::BabyBearD1Width16Bus1(air) => Self::BabyBearD1Width16Bus1(air.clone()),
            Self::BabyBearD1Width16Bus5(air) => Self::BabyBearD1Width16Bus5(air.clone()),
            Self::BabyBearD4Width16(air) => Self::BabyBearD4Width16(air.clone()),
            Self::BabyBearD4Width24(air) => Self::BabyBearD4Width24(air.clone()),
            Self::KoalaBearD1Width16Bus1(air) => Self::KoalaBearD1Width16Bus1(air.clone()),
            Self::KoalaBearD1Width16Bus5(air) => Self::KoalaBearD1Width16Bus5(air.clone()),
            Self::KoalaBearD4Width16(air) => Self::KoalaBearD4Width16(air.clone()),
            Self::KoalaBearD4Width24(air) => Self::KoalaBearD4Width24(air.clone()),
            Self::GoldilocksD2Width8(air) => Self::GoldilocksD2Width8(air.clone()),
        }
    }
}

pub(crate) struct Poseidon2AirWrapper<SC: StarkGenericConfig> {
    pub(crate) inner: Poseidon2AirWrapperInner,
    _phantom: core::marker::PhantomData<SC>,
}

impl<SC> BatchAir<SC> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
}

impl<SC: StarkGenericConfig> Clone for Poseidon2AirWrapper<SC> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: core::marker::PhantomData,
        }
    }
}

macro_rules! call_eval_variant {
    ($Params:ty, $Config:ty, $F:ty, $LL:ty, $AB:ty, $WITNESS_EXT:expr;
     $air:expr, $b:expr, $l:expr, $n:expr, $p:expr) => {
        eval_poseidon2_variant::<
            $Config,
            $F,
            $AB,
            $LL,
            { <$Params as Poseidon2Params>::D },
            { <$Params as Poseidon2Params>::WIDTH },
            { <$Params as Poseidon2Params>::WIDTH_EXT },
            { <$Params as Poseidon2Params>::RATE_EXT },
            { <$Params as Poseidon2Params>::CAPACITY_EXT },
            { <$Params as Poseidon2Params>::SBOX_DEGREE },
            { <$Params as Poseidon2Params>::SBOX_REGISTERS },
            { <$Params as Poseidon2Params>::HALF_FULL_ROUNDS },
            { <$Params as Poseidon2Params>::PARTIAL_ROUNDS },
            $WITNESS_EXT,
        >($air, $b, $l, $n, $p)
    };
}

macro_rules! eval_folder_inner {
    ($inner:expr, $builder:expr, $local:expr, $next:expr, $prep:expr;
     bb=$bb_ty:ty, kb=$kb_ty:ty, gl=$gl_ty:ty) => {
        match $inner {
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => unsafe {
                let b: &mut $bb_ty = transmute::<_, &mut $bb_ty>($builder);
                let l: &[<$bb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$bb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$bb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(BabyBearD1Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $bb_ty, 1;
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => unsafe {
                let b: &mut $bb_ty = transmute::<_, &mut $bb_ty>($builder);
                let l: &[<$bb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$bb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$bb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(BabyBearD1Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $bb_ty, 5;
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => unsafe {
                let b: &mut $bb_ty = transmute::<_, &mut $bb_ty>($builder);
                let l: &[<$bb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$bb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$bb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(BabyBearD4Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $bb_ty,
                    { <BabyBearD4Width16 as Poseidon2Params>::D };
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => unsafe {
                let b: &mut $bb_ty = transmute::<_, &mut $bb_ty>($builder);
                let l: &[<$bb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$bb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$bb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(BabyBearD4Width24, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $bb_ty,
                    { <BabyBearD4Width24 as Poseidon2Params>::D };
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => unsafe {
                let b: &mut $kb_ty = transmute::<_, &mut $kb_ty>($builder);
                let l: &[<$kb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$kb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$kb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(KoalaBearD1Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $kb_ty, 1;
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => unsafe {
                let b: &mut $kb_ty = transmute::<_, &mut $kb_ty>($builder);
                let l: &[<$kb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$kb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$kb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(KoalaBearD1Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $kb_ty, 5;
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => unsafe {
                let b: &mut $kb_ty = transmute::<_, &mut $kb_ty>($builder);
                let l: &[<$kb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$kb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$kb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(KoalaBearD4Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $kb_ty,
                    { <KoalaBearD4Width16 as Poseidon2Params>::D };
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => unsafe {
                let b: &mut $kb_ty = transmute::<_, &mut $kb_ty>($builder);
                let l: &[<$kb_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$kb_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$kb_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(KoalaBearD4Width24, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $kb_ty,
                    { <KoalaBearD4Width24 as Poseidon2Params>::D };
                    air.as_ref(), b, l, n, p);
            },
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => unsafe {
                let b: &mut $gl_ty = transmute::<_, &mut $gl_ty>($builder);
                let l: &[<$gl_ty as AirBuilder>::Var] = transmute($local);
                let n: &[<$gl_ty as AirBuilder>::Var] = transmute($next);
                let p: &[<$gl_ty as AirBuilder>::Var] = transmute($prep);
                call_eval_variant!(GoldilocksD2Width8, GoldilocksConfig, Goldilocks,
                    GenericPoseidon2LinearLayersGoldilocks, $gl_ty,
                    { <GoldilocksD2Width8 as Poseidon2Params>::D };
                    air.as_ref(), b, l, n, p);
            },
        }
    };
}

macro_rules! add_lookup_columns_inner {
    ($inner:expr) => {
        match $inner {
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => {
                LookupAir::add_lookup_columns(air.as_mut())
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! get_lookups_inner {
    ($inner:expr, $F:ty) => {
        match $inner {
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => unsafe {
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => unsafe {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => unsafe {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => unsafe {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => unsafe {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => unsafe {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => unsafe {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => unsafe {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => unsafe {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                let lookups = LookupAir::get_lookups(air.as_mut());
                core::mem::transmute(lookups)
            },
        }
    };
}

macro_rules! eval_symbolic_inner {
    ($inner:expr, $builder:expr, $F:ty) => {
        match $inner {
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<BabyBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<BabyBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<BabyBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => {
                assert_eq!(<$F>::from_u64(BABY_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<BabyBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<KoalaBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<KoalaBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<KoalaBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => {
                assert_eq!(<$F>::from_u64(KOALA_BEAR_MODULUS), <$F>::ZERO);
                unsafe {
                    let b: &mut SymbolicAirBuilder<KoalaBear> = core::mem::transmute($builder);
                    Air::eval(air.as_ref(), b);
                }
            }
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => unsafe {
                let b: &mut SymbolicAirBuilder<Goldilocks, BinomialExtensionField<Goldilocks, 2>> =
                    core::mem::transmute($builder);
                Air::eval(air.as_ref(), b);
            },
        }
    };
}

macro_rules! eval_verifier_inner {
    ($inner:expr, $builder:expr, $local:expr, $next:expr, $prep:expr; ab=$ab:ty) => {
        match $inner {
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => unsafe {
                call_eval_variant!(BabyBearD1Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $ab, 1;
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => unsafe {
                call_eval_variant!(BabyBearD1Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $ab, 5;
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => unsafe {
                call_eval_variant!(BabyBearD4Width16, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $ab,
                    { <BabyBearD4Width16 as Poseidon2Params>::D };
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => unsafe {
                call_eval_variant!(BabyBearD4Width24, BabyBearConfig, BabyBear,
                    GenericPoseidon2LinearLayersBabyBear, $ab,
                    { <BabyBearD4Width24 as Poseidon2Params>::D };
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => unsafe {
                call_eval_variant!(KoalaBearD1Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $ab, 1;
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => unsafe {
                call_eval_variant!(KoalaBearD1Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $ab, 5;
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => unsafe {
                call_eval_variant!(KoalaBearD4Width16, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $ab,
                    { <KoalaBearD4Width16 as Poseidon2Params>::D };
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => unsafe {
                call_eval_variant!(KoalaBearD4Width24, KoalaBearConfig, KoalaBear,
                    GenericPoseidon2LinearLayersKoalaBear, $ab,
                    { <KoalaBearD4Width24 as Poseidon2Params>::D };
                    air.as_ref(), $builder, $local, $next, $prep);
            },
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => unsafe {
                call_eval_variant!(GoldilocksD2Width8, GoldilocksConfig, Goldilocks,
                    GenericPoseidon2LinearLayersGoldilocks, $ab,
                    { <GoldilocksD2Width8 as Poseidon2Params>::D };
                    air.as_ref(), $builder, $local, $next, $prep);
            },
        }
    };
}

macro_rules! preprocessed_trace_inner {
    ($inner:expr, $SC:ty) => {
        match $inner {
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(air) => {
                assert_eq!(Val::<$SC>::from_u64(BABY_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<BabyBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<BabyBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(air) => {
                assert_eq!(Val::<$SC>::from_u64(BABY_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<BabyBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<BabyBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::BabyBearD4Width16(air) => {
                assert_eq!(Val::<$SC>::from_u64(BABY_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<BabyBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<BabyBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::BabyBearD4Width24(air) => {
                assert_eq!(Val::<$SC>::from_u64(BABY_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<BabyBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<BabyBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(air) => {
                assert_eq!(Val::<$SC>::from_u64(KOALA_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<KoalaBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<KoalaBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(air) => {
                assert_eq!(Val::<$SC>::from_u64(KOALA_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<KoalaBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<KoalaBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width16(air) => {
                assert_eq!(Val::<$SC>::from_u64(KOALA_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<KoalaBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<KoalaBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::KoalaBearD4Width24(air) => {
                assert_eq!(Val::<$SC>::from_u64(KOALA_BEAR_MODULUS), Val::<$SC>::ZERO);
                let p = BaseAir::<KoalaBear>::preprocessed_trace(air.as_ref())?;
                Some(unsafe { transmute::<RowMajorMatrix<KoalaBear>, RowMajorMatrix<Val<$SC>>>(p) })
            }
            Poseidon2AirWrapperInner::GoldilocksD2Width8(air) => {
                let p = BaseAir::<Goldilocks>::preprocessed_trace(air.as_ref())?;
                Some(unsafe {
                    transmute::<RowMajorMatrix<Goldilocks>, RowMajorMatrix<Val<$SC>>>(p)
                })
            }
        }
    };
}

impl<SC> BaseAir<Val<SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val<SC>>> {
        preprocessed_trace_inner!(&self.inner, SC)
    }
}

impl<SC> Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
{
    fn eval(&self, builder: &mut SymbolicAirBuilder<Val<SC>, SC::Challenge>) {
        eval_symbolic_inner!(&self.inner, builder, Val<SC>);
    }
}

impl<SC> LookupAir<Val<SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        add_lookup_columns_inner!(&mut self.inner)
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn get_lookups(
        &mut self,
    ) -> Vec<Lookup<<SymbolicAirBuilder<Val<SC>, SC::Challenge> as AirBuilder>::F>> {
        get_lookups_inner!(&mut self.inner, Val<SC>)
    }
}

impl<F: Field> BaseAir<F> for Poseidon2AirWrapperInner {
    fn width(&self) -> usize {
        match self {
            Self::BabyBearD1Width16Bus1(a) => BaseAir::<BabyBear>::width(a.as_ref()),
            Self::BabyBearD1Width16Bus5(a) => BaseAir::<BabyBear>::width(a.as_ref()),
            Self::BabyBearD4Width16(a) => BaseAir::<BabyBear>::width(a.as_ref()),
            Self::BabyBearD4Width24(a) => BaseAir::<BabyBear>::width(a.as_ref()),
            Self::KoalaBearD1Width16Bus1(a) => BaseAir::<KoalaBear>::width(a.as_ref()),
            Self::KoalaBearD1Width16Bus5(a) => BaseAir::<KoalaBear>::width(a.as_ref()),
            Self::KoalaBearD4Width16(a) => BaseAir::<KoalaBear>::width(a.as_ref()),
            Self::KoalaBearD4Width24(a) => BaseAir::<KoalaBear>::width(a.as_ref()),
            Self::GoldilocksD2Width8(a) => BaseAir::<Goldilocks>::width(a.as_ref()),
        }
    }
}

impl<F, EF> Air<SymbolicAirBuilder<F, EF>> for Poseidon2AirWrapperInner
where
    F: Field + PrimeField64,
    EF: ExtensionField<F>,
    SymbolicExpressionExt<F, EF>: Algebra<SymbolicExpression<F>>,
{
    fn eval(&self, builder: &mut SymbolicAirBuilder<F, EF>) {
        eval_symbolic_inner!(self, builder, F);
    }
}

impl<F> LookupAir<F> for Poseidon2AirWrapperInner
where
    F: Field + PrimeField64,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        add_lookup_columns_inner!(self)
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        get_lookups_inner!(self, F)
    }
}

pub fn poseidon2_verifier_air_from_config(config: Poseidon2Config) -> Poseidon2AirWrapperInner {
    Poseidon2Prover::air_wrapper_for_config(config)
}

pub(crate) unsafe fn eval_poseidon2_variant<
    SC,
    F: PrimeField,
    AB: AirBuilder,
    LinearLayers,
    const D: usize,
    const WIDTH: usize,
    const WIDTH_EXT: usize,
    const RATE_EXT: usize,
    const CAPACITY_EXT: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const WITNESS_EXT_D: usize,
>(
    air: &Poseidon2CircuitAir<
        F,
        LinearLayers,
        D,
        WIDTH,
        WIDTH_EXT,
        RATE_EXT,
        CAPACITY_EXT,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        WITNESS_EXT_D,
    >,
    builder: &mut AB,
    local_slice: &[<AB as AirBuilder>::Var],
    next_slice: &[<AB as AirBuilder>::Var],
    next_preprocessed_slice: &[<AB as AirBuilder>::Var],
) where
    SC: StarkGenericConfig,
    Val<SC>: StarkField + PrimeField,
    AB::F: PrimeField,
    LinearLayers: p3_poseidon2::GenericPoseidon2LinearLayers<WIDTH>,
{
    unsafe {
        type Cols<
            T,
            const W: usize,
            const SD: u64,
            const SR: usize,
            const HFR: usize,
            const PR: usize,
        > = p3_poseidon2_circuit_air::Poseidon2CircuitCols<
            T,
            p3_poseidon2_air::Poseidon2Cols<T, W, SD, SR, HFR, PR>,
        >;

        let local_var: &Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = &*(local_slice.as_ptr()
            as *const Cols<
                AB::Var,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >);

        let next_var: &Cols<
            AB::Var,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = &*(next_slice.as_ptr()
            as *const Cols<
                AB::Var,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >);

        eval_unchecked::<
            F,
            AB,
            LinearLayers,
            D,
            WIDTH,
            WIDTH_EXT,
            RATE_EXT,
            CAPACITY_EXT,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            WITNESS_EXT_D,
        >(air, builder, local_var, next_var, next_preprocessed_slice);
    }
}

impl<'a, SC> Air<ProverConstraintFolder<'a, SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField + PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn eval(&self, builder: &mut ProverConstraintFolder<'a, SC>) {
        let main = builder.main();
        let local_slice = main.current_slice();
        let next_slice = main.next_slice();
        let preprocessed = *builder.preprocessed();
        let next_preprocessed_slice = preprocessed.next_slice();

        eval_folder_inner!(
            &self.inner, builder, local_slice, next_slice, next_preprocessed_slice;
            bb=ProverConstraintFolder<'a, BabyBearConfig>,
            kb=ProverConstraintFolder<'a, KoalaBearConfig>,
            gl=ProverConstraintFolder<'a, GoldilocksConfig>
        );
    }
}

impl<'a, SC> Air<ProverConstraintFolderWithLookups<'a, SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField + PrimeField,
{
    fn eval(&self, builder: &mut ProverConstraintFolderWithLookups<'a, SC>) {
        let main = builder.main();
        let local_slice = main.current_slice();
        let next_slice = main.next_slice();
        let preprocessed = *builder.preprocessed();
        let next_preprocessed_slice = preprocessed.next_slice();

        eval_folder_inner!(
            &self.inner, builder, local_slice, next_slice, next_preprocessed_slice;
            bb=ProverConstraintFolderWithLookups<'a, BabyBearConfig>,
            kb=ProverConstraintFolderWithLookups<'a, KoalaBearConfig>,
            gl=ProverConstraintFolderWithLookups<'a, GoldilocksConfig>
        );
    }
}

impl<'a, SC> Air<VerifierConstraintFolder<'a, SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField + PrimeField,
{
    fn eval(&self, builder: &mut VerifierConstraintFolder<'a, SC>) {
        let main = builder.main();
        let local_slice = main.current_slice();
        let next_slice = main.next_slice();
        let preprocessed = *builder.preprocessed();
        let next_preprocessed_slice = preprocessed.next_slice();

        eval_verifier_inner!(
            &self.inner, builder, &local_slice, &next_slice, &next_preprocessed_slice;
            ab=VerifierConstraintFolder<'a, SC>
        );
    }
}

impl<'a, SC> Air<VerifierConstraintFolderWithLookups<'a, SC>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField + PrimeField,
{
    fn eval(&self, builder: &mut VerifierConstraintFolderWithLookups<'a, SC>) {
        let main = builder.main();
        let local_slice = main.current_slice();
        let next_slice = main.next_slice();
        let preprocessed = *builder.preprocessed();
        let next_preprocessed_slice = preprocessed.next_slice();

        eval_verifier_inner!(
            &self.inner, builder, &local_slice, &next_slice, &next_preprocessed_slice;
            ab=VerifierConstraintFolderWithLookups<'a, SC>
        );
    }
}

#[cfg(debug_assertions)]
impl<'a, SC> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>> for Poseidon2AirWrapper<SC>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField + PrimeField,
{
    fn eval(&self, builder: &mut DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>) {
        let main = builder.main();
        let local_slice = main.current_slice();
        let next_slice = main.next_slice();
        let preprocessed = *builder.preprocessed();
        let next_preprocessed_slice = preprocessed.next_slice();

        eval_verifier_inner!(
            &self.inner, builder, &local_slice, &next_slice, &next_preprocessed_slice;
            ab=DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>
        );
    }
}

#[derive(Clone)]
pub struct Poseidon2Prover {
    config: Poseidon2Config,
}

unsafe impl Send for Poseidon2Prover {}
unsafe impl Sync for Poseidon2Prover {}

impl Poseidon2Prover {
    #[allow(dead_code)] // Used by tests / future hooks; batch path uses `self.config` directly.
    pub(crate) const fn config(&self) -> Poseidon2Config {
        self.config
    }

    pub(crate) fn poseidon2_op_type(&self) -> NpoTypeId {
        NpoTypeId::poseidon2_perm(self.config)
    }

    pub const fn new(
        config: Poseidon2Config,
        _profile: crate::constraint_profile::ConstraintProfile,
    ) -> Self {
        Self { config }
    }

    pub(crate) fn air_wrapper_for_config(config: Poseidon2Config) -> Poseidon2AirWrapperInner {
        match config {
            Poseidon2Config::BabyBearD1Width16 => Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(
                Box::new(BabyBearD1Width16::default_air()),
            ),
            Poseidon2Config::BabyBearD4Width16 => Poseidon2AirWrapperInner::BabyBearD4Width16(
                Box::new(BabyBearD4Width16::default_air()),
            ),
            Poseidon2Config::BabyBearD4Width24 => Poseidon2AirWrapperInner::BabyBearD4Width24(
                Box::new(BabyBearD4Width24::default_air()),
            ),
            Poseidon2Config::KoalaBearD1Width16 => {
                Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(Box::new(
                    KoalaBearD1Width16::default_air(),
                ))
            }
            Poseidon2Config::KoalaBearD4Width16 => Poseidon2AirWrapperInner::KoalaBearD4Width16(
                Box::new(KoalaBearD4Width16::default_air()),
            ),
            Poseidon2Config::KoalaBearD4Width24 => Poseidon2AirWrapperInner::KoalaBearD4Width24(
                Box::new(KoalaBearD4Width24::default_air()),
            ),
            Poseidon2Config::GoldilocksD2Width8 => Poseidon2AirWrapperInner::GoldilocksD2Width8(
                Box::new(goldilocks_d2_width8_default_air()),
            ),
        }
    }

    fn air_wrapper_for_config_with_preprocessed<F: Field>(
        config: Poseidon2Config,
        preprocessed: Vec<F>,
        min_height: usize,
        circuit_extension_degree: u32,
    ) -> Option<Poseidon2AirWrapperInner> {
        let inner = match config {
            Poseidon2Config::BabyBearD1Width16 => {
                assert!(F::from_u64(BABY_BEAR_MODULUS) == F::ZERO);
                let prep = unsafe { transmute::<Vec<F>, Vec<BabyBear>>(preprocessed) };
                match poseidon_d1_witness_bus_dim(circuit_extension_degree)? {
                    1 => Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(Box::new(
                        BabyBearD1Width16::default_air_with_preprocessed(prep, min_height),
                    )),
                    5 => Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(Box::new(
                        BabyBearD1Width16::default_air_with_preprocessed_witness_bus5(
                            prep, min_height,
                        ),
                    )),
                    _ => unreachable!(),
                }
            }
            Poseidon2Config::BabyBearD4Width16 => {
                assert!(F::from_u64(BABY_BEAR_MODULUS) == F::ZERO);
                Poseidon2AirWrapperInner::BabyBearD4Width16(Box::new(
                    BabyBearD4Width16::default_air_with_preprocessed(
                        unsafe { transmute::<Vec<F>, Vec<BabyBear>>(preprocessed) },
                        min_height,
                    ),
                ))
            }
            Poseidon2Config::BabyBearD4Width24 => {
                assert!(F::from_u64(BABY_BEAR_MODULUS) == F::ZERO);
                Poseidon2AirWrapperInner::BabyBearD4Width24(Box::new(
                    BabyBearD4Width24::default_air_with_preprocessed(
                        unsafe { transmute::<Vec<F>, Vec<BabyBear>>(preprocessed) },
                        min_height,
                    ),
                ))
            }
            Poseidon2Config::KoalaBearD1Width16 => {
                assert!(F::from_u64(KOALA_BEAR_MODULUS) == F::ZERO);
                let prep = unsafe { transmute::<Vec<F>, Vec<KoalaBear>>(preprocessed) };
                match poseidon_d1_witness_bus_dim(circuit_extension_degree)? {
                    1 => Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(Box::new(
                        KoalaBearD1Width16::default_air_with_preprocessed(prep, min_height),
                    )),
                    5 => Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(Box::new(
                        KoalaBearD1Width16::default_air_with_preprocessed_witness_bus5(
                            prep, min_height,
                        ),
                    )),
                    _ => unreachable!(),
                }
            }
            Poseidon2Config::KoalaBearD4Width16 => {
                assert!(F::from_u64(KOALA_BEAR_MODULUS) == F::ZERO);
                Poseidon2AirWrapperInner::KoalaBearD4Width16(Box::new(
                    KoalaBearD4Width16::default_air_with_preprocessed(
                        unsafe { transmute::<Vec<F>, Vec<KoalaBear>>(preprocessed) },
                        min_height,
                    ),
                ))
            }
            Poseidon2Config::KoalaBearD4Width24 => {
                assert!(F::from_u64(KOALA_BEAR_MODULUS) == F::ZERO);
                Poseidon2AirWrapperInner::KoalaBearD4Width24(Box::new(
                    KoalaBearD4Width24::default_air_with_preprocessed(
                        unsafe { transmute::<Vec<F>, Vec<KoalaBear>>(preprocessed) },
                        min_height,
                    ),
                ))
            }
            Poseidon2Config::GoldilocksD2Width8 => Poseidon2AirWrapperInner::GoldilocksD2Width8(
                Box::new(goldilocks_d2_width8_default_air_with_preprocessed(
                    unsafe { transmute::<Vec<F>, Vec<Goldilocks>>(preprocessed) },
                    min_height,
                )),
            ),
        };
        Some(inner)
    }

    pub fn wrapper_from_config_with_preprocessed<SC>(
        &self,
        preprocessed: Vec<Val<SC>>,
        min_height: usize,
        circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let inner = Self::air_wrapper_for_config_with_preprocessed::<Val<SC>>(
            self.config,
            preprocessed,
            min_height,
            circuit_extension_degree,
        )?;
        Some(DynamicAirEntry::new(Box::new(Poseidon2AirWrapper {
            inner,
            _phantom: core::marker::PhantomData::<SC>,
        })))
    }

    pub const fn preprocessed_width_from_config(&self) -> usize {
        match self.config {
            Poseidon2Config::BabyBearD1Width16 => {
                Poseidon2CircuitAirBabyBearD1Width16::preprocessed_width()
            }
            Poseidon2Config::BabyBearD4Width16 => {
                Poseidon2CircuitAirBabyBearD4Width16::preprocessed_width()
            }
            Poseidon2Config::BabyBearD4Width24 => {
                Poseidon2CircuitAirBabyBearD4Width24::preprocessed_width()
            }
            Poseidon2Config::KoalaBearD1Width16 => {
                Poseidon2CircuitAirKoalaBearD1Width16::preprocessed_width()
            }
            Poseidon2Config::KoalaBearD4Width16 => {
                Poseidon2CircuitAirKoalaBearD4Width16::preprocessed_width()
            }
            Poseidon2Config::KoalaBearD4Width24 => {
                Poseidon2CircuitAirKoalaBearD4Width24::preprocessed_width()
            }
            Poseidon2Config::GoldilocksD2Width8 => {
                Poseidon2CircuitAirGoldilocksD2Width8::preprocessed_width()
            }
        }
    }

    fn batch_instance_from_traces<SC, CF>(
        &self,
        _config: &SC,
        packing: &TablePacking,
        traces: &Traces<CF>,
    ) -> Option<BatchTableInstance<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        CF: Field + ExtensionField<Val<SC>> + BasedVectorSpace<Val<SC>>,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let t = traces.non_primitive_trace::<Poseidon2Trace<Val<SC>>>(
            &NpoTypeId::poseidon2_perm(self.config),
        )?;

        let rows = t.total_rows();
        if rows == 0 {
            return None;
        }

        let min_height = packing.min_trace_height();
        let witness_ctl_scale = <CF as BasedVectorSpace<Val<SC>>>::DIMENSION as u32;
        self.batch_instance_base_impl::<SC>(t, min_height, witness_ctl_scale)
    }

    fn batch_instance_base_impl<SC>(
        &self,
        t: &Poseidon2Trace<Val<SC>>,
        min_height: usize,
        witness_ctl_scale: u32,
    ) -> Option<BatchTableInstance<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let cfg = self.config;
        let rows = t.total_rows();

        // Pad logical ops to the larger of (next power-of-two of row count) and `min_height`.
        let padded_rows = rows.next_power_of_two().max(min_height.next_power_of_two());
        let width = cfg.width();
        let width_ext = cfg.width_ext();
        let rate_ext = cfg.rate_ext();
        // Must match `Poseidon2CircuitAir::preprocessed_trace`: first padded row is a sponge
        // chain boundary (`new_start` in preprocessed) with zero state. Duplicating the last real
        // row would leave non-zero capacity inputs and break compact D=1 constraints that assert
        // zero capacity on sponge `new_start` transitions.
        let pad_filler = Poseidon2CircuitRow {
            new_start: true,
            merkle_path: false,
            mmcs_bit: false,
            mmcs_index_sum: Val::<SC>::ZERO,
            input_values: Val::<SC>::zero_vec(width),
            in_ctl: vec![false; width_ext],
            input_indices: vec![0; width_ext],
            out_ctl: vec![false; rate_ext],
            output_indices: vec![0; rate_ext],
            mmcs_index_sum_idx: 0,
            mmcs_ctl_enabled: false,
        };
        let mut padded_ops = t.operations.clone();
        padded_ops.resize(padded_rows, pad_filler);

        let (air, matrix) = match cfg {
            Poseidon2Config::BabyBearD1Width16 => {
                let constants = BabyBearD1Width16::round_constants();
                let wbus = poseidon_d1_witness_bus_dim(witness_ctl_scale)?;
                let preprocessed = extract_preprocessed_from_operations::<16, 8, BabyBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    1,
                );
                let (inner, matrix_f) = match wbus {
                    1 => {
                        let air = BabyBearD1Width16::default_air_with_preprocessed(
                            preprocessed,
                            min_height,
                        );
                        let ops: Vec<Poseidon2CircuitRow<BabyBear>> =
                            unsafe { transmute(padded_ops) };
                        let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                        (
                            Poseidon2AirWrapperInner::BabyBearD1Width16Bus1(Box::new(air)),
                            matrix_f,
                        )
                    }
                    5 => {
                        let air = BabyBearD1Width16::default_air_with_preprocessed_witness_bus5(
                            preprocessed,
                            min_height,
                        );
                        let ops: Vec<Poseidon2CircuitRow<BabyBear>> =
                            unsafe { transmute(padded_ops) };
                        let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                        (
                            Poseidon2AirWrapperInner::BabyBearD1Width16Bus5(Box::new(air)),
                            matrix_f,
                        )
                    }
                    _ => unreachable!(),
                };
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner,
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::BabyBearD4Width16 => {
                let constants = BabyBearD4Width16::round_constants();
                let preprocessed = extract_preprocessed_from_operations::<4, 2, BabyBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    cfg.d(),
                );
                let air =
                    BabyBearD4Width16::default_air_with_preprocessed(preprocessed, min_height);
                let ops: Vec<Poseidon2CircuitRow<BabyBear>> = unsafe { transmute(padded_ops) };
                let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner: Poseidon2AirWrapperInner::BabyBearD4Width16(Box::new(air)),
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::BabyBearD4Width24 => {
                let constants = BabyBearD4Width24::round_constants();
                let preprocessed = extract_preprocessed_from_operations::<6, 4, BabyBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    cfg.d(),
                );
                let air =
                    BabyBearD4Width24::default_air_with_preprocessed(preprocessed, min_height);
                let ops: Vec<Poseidon2CircuitRow<BabyBear>> = unsafe { transmute(padded_ops) };
                let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner: Poseidon2AirWrapperInner::BabyBearD4Width24(Box::new(air)),
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::KoalaBearD1Width16 => {
                let constants = KoalaBearD1Width16::round_constants();
                let wbus = poseidon_d1_witness_bus_dim(witness_ctl_scale)?;
                let preprocessed = extract_preprocessed_from_operations::<16, 8, KoalaBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    1,
                );
                let (inner, matrix_f) = match wbus {
                    1 => {
                        let air = KoalaBearD1Width16::default_air_with_preprocessed(
                            preprocessed,
                            min_height,
                        );
                        let ops: Vec<Poseidon2CircuitRow<KoalaBear>> =
                            unsafe { transmute(padded_ops) };
                        let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                        (
                            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus1(Box::new(air)),
                            matrix_f,
                        )
                    }
                    5 => {
                        let air = KoalaBearD1Width16::default_air_with_preprocessed_witness_bus5(
                            preprocessed,
                            min_height,
                        );
                        let ops: Vec<Poseidon2CircuitRow<KoalaBear>> =
                            unsafe { transmute(padded_ops) };
                        let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                        (
                            Poseidon2AirWrapperInner::KoalaBearD1Width16Bus5(Box::new(air)),
                            matrix_f,
                        )
                    }
                    _ => unreachable!(),
                };
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner,
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::KoalaBearD4Width16 => {
                let constants = KoalaBearD4Width16::round_constants();
                let preprocessed = extract_preprocessed_from_operations::<4, 2, KoalaBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    cfg.d(),
                );
                let air =
                    KoalaBearD4Width16::default_air_with_preprocessed(preprocessed, min_height);
                let ops: Vec<Poseidon2CircuitRow<KoalaBear>> = unsafe { transmute(padded_ops) };
                let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner: Poseidon2AirWrapperInner::KoalaBearD4Width16(Box::new(air)),
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::KoalaBearD4Width24 => {
                let constants = KoalaBearD4Width24::round_constants();
                let preprocessed = extract_preprocessed_from_operations::<6, 4, KoalaBear, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    cfg.d(),
                );
                let air =
                    KoalaBearD4Width24::default_air_with_preprocessed(preprocessed, min_height);
                let ops: Vec<Poseidon2CircuitRow<KoalaBear>> = unsafe { transmute(padded_ops) };
                let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner: Poseidon2AirWrapperInner::KoalaBearD4Width24(Box::new(air)),
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
            Poseidon2Config::GoldilocksD2Width8 => {
                let constants = goldilocks_d2_width8_round_constants();
                let preprocessed = extract_preprocessed_from_operations::<4, 2, Goldilocks, Val<SC>>(
                    &t.operations,
                    witness_ctl_scale,
                    cfg.d(),
                );
                let air =
                    goldilocks_d2_width8_default_air_with_preprocessed(preprocessed, min_height);
                let ops: Vec<Poseidon2CircuitRow<Goldilocks>> = unsafe { transmute(padded_ops) };
                let matrix_f = air.generate_trace_rows(&ops, &constants, 0);
                let matrix: RowMajorMatrix<Val<SC>> = unsafe { transmute(matrix_f) };
                (
                    Poseidon2AirWrapper {
                        inner: Poseidon2AirWrapperInner::GoldilocksD2Width8(Box::new(air)),
                        _phantom: core::marker::PhantomData::<SC>,
                    },
                    matrix,
                )
            }
        };

        Some(BatchTableInstance {
            op_type: NpoTypeId::poseidon2_perm(self.config),
            air: DynamicAirEntry::new(Box::new(air)),
            trace: matrix,
            public_values: Vec::new(),
            rows: padded_rows,
            lanes: 1,
        })
    }
}

impl<SC> TableProver<SC> for Poseidon2Prover
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<4>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        self.poseidon2_op_type()
    }

    fn batch_instance_d1(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>> {
        self.batch_instance_from_traces::<SC, Val<SC>>(config, packing, traces)
    }

    fn batch_instance_d2(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<BinomialExtensionField<Val<SC>, 2>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    fn batch_instance_d4(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 4>>,
    ) -> Option<BatchTableInstance<SC>> {
        self.batch_instance_from_traces::<SC, BinomialExtensionField<Val<SC>, 4>>(
            config, packing, traces,
        )
    }

    fn batch_instance_d5(
        &self,
        _config: &SC,
        packing: &TablePacking,
        traces: &Traces<QuinticTrinomialExtensionField<Val<SC>>>,
    ) -> Option<BatchTableInstance<SC>> {
        let t = traces.non_primitive_trace::<Poseidon2Trace<Val<SC>>>(
            &NpoTypeId::poseidon2_perm(self.config),
        )?;
        let rows = t.total_rows();
        if rows == 0 {
            return None;
        }
        let min_height = packing.min_trace_height();
        self.batch_instance_base_impl::<SC>(t, min_height, 5)
    }

    fn batch_instance_d6(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 6>>,
    ) -> Option<BatchTableInstance<SC>> {
        let _ = (config, packing, traces);
        None
    }

    fn batch_instance_d8(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 8>>,
    ) -> Option<BatchTableInstance<SC>> {
        let _ = (config, packing, traces);
        None
    }

    fn batch_air_from_table_entry(
        &self,
        _config: &SC,
        _degree: usize,
        circuit_extension_degree: u32,
        _table_entry: &NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String> {
        self.wrapper_from_config_with_preprocessed(Vec::new(), 1, circuit_extension_degree)
            .ok_or_else(|| {
                format!(
                    "unsupported witness bus dimension {} for Poseidon2 config {:?}",
                    circuit_extension_degree, self.config
                )
            })
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        _lanes: usize,
        circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        self.wrapper_from_config_with_preprocessed(
            committed_prep,
            min_height,
            circuit_extension_degree,
        )
    }
}

/// Build the expanded Poseidon2 circuit-AIR table instance without invoking
/// the batch STARK/FRI prover.
///
/// WHIR-native proving reuses this as the semantic bridge from compact
/// `Poseidon2Trace` rows to the full round-state matrix and AIR wrapper used by
/// the FRI-backed batch prover.
#[allow(dead_code)]
pub(crate) fn poseidon2_whir_native_batch_instance<SC>(
    config: Poseidon2Config,
    trace: &Poseidon2Trace<Val<SC>>,
    min_height: usize,
    witness_ctl_scale: u32,
) -> Option<BatchTableInstance<SC>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    Poseidon2Prover::new(config, ConstraintProfile::Standard).batch_instance_base_impl::<SC>(
        trace,
        min_height,
        witness_ctl_scale,
    )
}
pub struct Poseidon2ProverD2(pub(crate) Poseidon2Prover);

impl Poseidon2ProverD2 {
    pub const fn new(
        config: Poseidon2Config,
        profile: crate::constraint_profile::ConstraintProfile,
    ) -> Self {
        Self(Poseidon2Prover::new(config, profile))
    }
}

impl<SC> TableProver<SC> for Poseidon2ProverD2
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<2>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        self.0.poseidon2_op_type()
    }

    fn batch_instance_d1(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    fn batch_instance_d2(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 2>>,
    ) -> Option<BatchTableInstance<SC>> {
        self.0
            .batch_instance_from_traces::<SC, BinomialExtensionField<Val<SC>, 2>>(
                config, packing, traces,
            )
    }

    fn batch_instance_d4(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<BinomialExtensionField<Val<SC>, 4>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    fn batch_instance_d6(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<BinomialExtensionField<Val<SC>, 6>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    fn batch_instance_d8(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<BinomialExtensionField<Val<SC>, 8>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    fn batch_air_from_table_entry(
        &self,
        _config: &SC,
        _degree: usize,
        circuit_extension_degree: u32,
        _table_entry: &NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String> {
        self.0
            .wrapper_from_config_with_preprocessed(Vec::new(), 1, circuit_extension_degree)
            .ok_or_else(|| {
                format!(
                    "unsupported witness bus dimension {} for Poseidon2 config {:?}",
                    circuit_extension_degree, self.0.config
                )
            })
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        _lanes: usize,
        circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        self.0.wrapper_from_config_with_preprocessed(
            committed_prep,
            min_height,
            circuit_extension_degree,
        )
    }
}

/// Shared helper implementing Poseidon2-specific preprocessing on generic preprocessed columns.
fn poseidon2_preprocess_for_prover<F, ExtF, const D: usize>(
    preprocessed: &mut PreprocessedColumns<ExtF, D>,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    ExtF: ExtensionField<F>,
{
    let neg_one = F::NEG_ONE;

    // Phase 1: scan Poseidon2 preprocessed data to count mmcs_index_sum conditional reads,
    // and update `ext_reads` accordingly. This must happen before computing multiplicities.
    for (op_type, prep) in preprocessed.non_primitive.iter() {
        let op_str = op_type.as_str();
        if !op_str.starts_with("poseidon2_perm/") {
            continue;
        }
        let rest = op_str
            .strip_prefix("poseidon2_perm/")
            .ok_or(CircuitError::InvalidPreprocessedValues)?;
        let p2_cfg = Poseidon2Config::from_variant_name(rest)
            .ok_or(CircuitError::InvalidPreprocessedValues)?;
        let w_ext = p2_cfg.width_ext();
        let r_ext = p2_cfg.rate_ext();
        let prep_row_width = poseidon2_preprocessed_row_width_for_air(p2_cfg.d(), w_ext, r_ext);

        let prep_base: Vec<F> = prep
            .iter()
            .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        if !prep_base.len().is_multiple_of(prep_row_width) {
            return Err(CircuitError::InvalidPreprocessedValues);
        }

        let num_rows = prep_base.len() / prep_row_width;
        let trace_height = num_rows.next_power_of_two();
        let has_padding = trace_height > num_rows;
        let compact = poseidon2_uses_compact_d1_preprocessed(p2_cfg.d(), w_ext, r_ext);
        let tail = if compact {
            poseidon2_d1_compact_preprocessed_header_cols(r_ext) + w_ext + r_ext + r_ext
        } else {
            poseidon2_preprocessed_row_width(w_ext, r_ext) - 4
        };

        for row_idx in 0..num_rows {
            let row_start = row_idx * prep_row_width;
            let mmcs_flag_off = row_start + tail + 1;
            let current_mmcs_merkle_flag = prep_base[mmcs_flag_off];

            // Check if next row exists and has new_start = 1.
            // The Poseidon2 AIR pads the trace and sets new_start = 1 in the first
            // padding row (only if padding exists), so the last real row can trigger a
            // lookup if its mmcs_merkle_flag = 1 and there is padding.
            let next_new_start = if row_idx + 1 < num_rows {
                let next_start = (row_idx + 1) * prep_row_width;
                prep_base[next_start + tail + 2]
            } else if has_padding {
                F::ONE
            } else {
                prep_base[tail + 2]
            };

            let multiplicity = current_mmcs_merkle_flag * next_new_start;
            if multiplicity != F::ZERO {
                let mmcs_idx_u64 = F::as_canonical_u64(&prep_base[row_start + tail]);
                let mmcs_witness_idx = (mmcs_idx_u64 as usize) / D;

                if mmcs_witness_idx >= preprocessed.ext_reads.len() {
                    preprocessed.ext_reads.resize(mmcs_witness_idx + 1, 0);
                }
                preprocessed.ext_reads[mmcs_witness_idx] += 1;
            }
        }
    }

    // Phase 2: update Poseidon2 out_ctl values in the base-field preprocessed data.
    //
    // Poseidon2 duplicate creators (from optimizer witness_rewrite deduplication)
    // are recorded in plugin-owned metadata under this op_type. For those, out_ctl = -1
    // (reader contribution). For first-occurrence creators, out_ctl = +ext_reads[wid].
    let mut non_primitive_base: NonPrimitivePreprocessedMap<F> = HashMap::new();
    for (op_type, prep) in preprocessed.non_primitive.iter() {
        let op_str = op_type.as_str();
        if !op_str.starts_with("poseidon2_perm/") {
            continue;
        }
        let rest = op_str
            .strip_prefix("poseidon2_perm/")
            .ok_or(CircuitError::InvalidPreprocessedValues)?;
        let p2_cfg = Poseidon2Config::from_variant_name(rest)
            .ok_or(CircuitError::InvalidPreprocessedValues)?;
        let w_ext = p2_cfg.width_ext();
        let r_ext = p2_cfg.rate_ext();
        let prep_row_width = poseidon2_preprocessed_row_width_for_air(p2_cfg.d(), w_ext, r_ext);

        let dup_wids = preprocessed.dup_npo_outputs.get(op_type);

        let mut prep_base: Vec<F> = prep
            .iter()
            .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
            .collect::<Result<Vec<_>, CircuitError>>()?;

        if !prep_base.len().is_multiple_of(prep_row_width) {
            return Err(CircuitError::InvalidPreprocessedValues);
        }

        let num_rows = prep_base.len() / prep_row_width;
        let compact = poseidon2_uses_compact_d1_preprocessed(p2_cfg.d(), w_ext, r_ext);

        for row_idx in 0..num_rows {
            let row_start = row_idx * prep_row_width;
            let out_base = if compact {
                row_start + poseidon2_d1_compact_preprocessed_header_cols(r_ext) + w_ext
            } else {
                row_start + w_ext * size_of::<Poseidon2PrepInputLimb<u8>>()
            };
            for j in 0..r_ext {
                let (o0, ctl_off) = if compact {
                    (out_base + j, out_base + r_ext + j)
                } else {
                    let o = out_base + j * 2;
                    (o, o + 1)
                };
                let out_ctl = prep_base[ctl_off];
                if out_ctl != F::ZERO {
                    let idx = prep_base[o0];
                    let out_wid = F::as_canonical_u64(&idx) as usize / D;
                    let is_dup = dup_wids
                        .and_then(|d| d.get(out_wid).copied())
                        .unwrap_or(false);
                    prep_base[ctl_off] = if is_dup {
                        neg_one
                    } else {
                        let n_reads = preprocessed.ext_reads.get(out_wid).copied().unwrap_or(0);
                        F::from_u32(n_reads)
                    };
                }
            }
        }

        non_primitive_base.insert(op_type.clone(), prep_base);
    }

    Ok(non_primitive_base)
}

/// Stateless plugin used for Poseidon2 preprocessing.
#[derive(Clone, Default)]
pub struct Poseidon2Preprocessor;

impl NpoPreprocessor<BabyBear> for Poseidon2Preprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn Any,
        preprocessed: &mut dyn Any,
    ) -> Result<NonPrimitivePreprocessedMap<BabyBear>, CircuitError> {
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<BabyBear, 1>>() {
            return poseidon2_preprocess_for_prover::<BabyBear, BabyBear, 1>(prep);
        }
        if let Some(prep) = preprocessed
            .downcast_mut::<PreprocessedColumns<BinomialExtensionField<BabyBear, 4>, 4>>()
        {
            return poseidon2_preprocess_for_prover::<
                BabyBear,
                BinomialExtensionField<BabyBear, 4>,
                4,
            >(prep);
        }
        Ok(NonPrimitivePreprocessedMap::new())
    }
}

impl NpoPreprocessor<KoalaBear> for Poseidon2Preprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn Any,
        preprocessed: &mut dyn Any,
    ) -> Result<NonPrimitivePreprocessedMap<KoalaBear>, CircuitError> {
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<KoalaBear, 1>>() {
            return poseidon2_preprocess_for_prover::<KoalaBear, KoalaBear, 1>(prep);
        }
        if let Some(prep) = preprocessed
            .downcast_mut::<PreprocessedColumns<BinomialExtensionField<KoalaBear, 4>, 4>>()
        {
            return poseidon2_preprocess_for_prover::<
                KoalaBear,
                BinomialExtensionField<KoalaBear, 4>,
                4,
            >(prep);
        }
        if let Some(prep) = preprocessed
            .downcast_mut::<PreprocessedColumns<QuinticTrinomialExtensionField<KoalaBear>, 5>>()
        {
            return poseidon2_preprocess_for_prover::<
                KoalaBear,
                QuinticTrinomialExtensionField<KoalaBear>,
                5,
            >(prep);
        }
        Ok(NonPrimitivePreprocessedMap::new())
    }
}

impl NpoPreprocessor<Goldilocks> for Poseidon2Preprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn Any,
        preprocessed: &mut dyn Any,
    ) -> Result<NonPrimitivePreprocessedMap<Goldilocks>, CircuitError> {
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<Goldilocks, 1>>() {
            return poseidon2_preprocess_for_prover::<Goldilocks, Goldilocks, 1>(prep);
        }
        if let Some(prep) = preprocessed
            .downcast_mut::<PreprocessedColumns<BinomialExtensionField<Goldilocks, 2>, 2>>()
        {
            return poseidon2_preprocess_for_prover::<
                Goldilocks,
                BinomialExtensionField<Goldilocks, 2>,
                2,
            >(prep);
        }
        Ok(NonPrimitivePreprocessedMap::new())
    }
}

/// Returns `Some(config)` when this Poseidon2 variant is supported for batch AIR building at
/// extension degree `D` (2 = Goldilocks, 4 = BabyBear / KoalaBear, 5 = KoalaBear quintic).
///
/// For `D = 5` only D=1 (base-field) configs are valid: the quintic challenger always operates
/// in the base field.
pub(crate) fn poseidon2_config_for_air_builder<const D: usize>(
    config: Poseidon2Config,
) -> Option<Poseidon2Config> {
    match D {
        2 => match config {
            Poseidon2Config::GoldilocksD2Width8 => Some(config),
            _ => None,
        },
        4 => match config {
            Poseidon2Config::BabyBearD1Width16
            | Poseidon2Config::BabyBearD4Width16
            | Poseidon2Config::BabyBearD4Width24
            | Poseidon2Config::KoalaBearD1Width16
            | Poseidon2Config::KoalaBearD4Width16
            | Poseidon2Config::KoalaBearD4Width24 => Some(config),
            _ => None,
        },
        5 => match config {
            Poseidon2Config::BabyBearD1Width16 | Poseidon2Config::KoalaBearD1Width16 => {
                Some(config)
            }
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn poseidon2_air_try_build<SC, const D: usize>(
    op_type: &NpoTypeId,
    prep_base: &[Val<SC>],
    min_height: usize,
    constraint_profile: ConstraintProfile,
) -> Option<(CircuitTableAir<SC, D>, usize)>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<D>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    let suffix = op_type.as_str().strip_prefix("poseidon2_perm/")?;
    let config = Poseidon2Config::from_variant_name(suffix)?;
    let config = poseidon2_config_for_air_builder::<D>(config)?;
    let prover = Poseidon2Prover::new(config, constraint_profile);
    let wrapper =
        prover.wrapper_from_config_with_preprocessed(prep_base.to_vec(), min_height, D as u32)?;
    let width = prover.preprocessed_width_from_config();
    let num_rows = prep_base.len().div_ceil(width);
    let degree = log2_ceil_usize(
        num_rows
            .next_power_of_two()
            .max(min_height.next_power_of_two()),
    );
    Some((CircuitTableAir::Dynamic(wrapper), degree))
}

/// Poseidon2 NPO AIR builder parameterized by extension degree `D`.
#[derive(Clone, Default)]
pub struct Poseidon2AirBuilder<const D: usize>;

impl<SC> NpoAirBuilder<SC, 2> for Poseidon2AirBuilder<2>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<2>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        _lanes: usize,
        constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, 2>, usize)> {
        poseidon2_air_try_build::<SC, 2>(op_type, prep_base, min_height, constraint_profile)
    }
}

impl<SC> NpoAirBuilder<SC, 4> for Poseidon2AirBuilder<4>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<4>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        _lanes: usize,
        constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, 4>, usize)> {
        poseidon2_air_try_build::<SC, 4>(op_type, prep_base, min_height, constraint_profile)
    }
}

impl<SC> NpoAirBuilder<SC, 5> for Poseidon2AirBuilder<5>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        _lanes: usize,
        constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, 5>, usize)> {
        let suffix = op_type.as_str().strip_prefix("poseidon2_perm/")?;
        let config = Poseidon2Config::from_variant_name(suffix)?;
        // For D=5 circuits the Poseidon2 permutation always operates in the base field
        // (the quintic challenger uses D=1 configs).
        let config = match config {
            Poseidon2Config::BabyBearD1Width16 | Poseidon2Config::KoalaBearD1Width16 => config,
            _ => return None,
        };
        let prover = Poseidon2Prover::new(config, constraint_profile);
        let wrapper =
            prover.wrapper_from_config_with_preprocessed(prep_base.to_vec(), min_height, 5)?;
        let width = prover.preprocessed_width_from_config();
        let num_rows = prep_base.len().div_ceil(width);
        let degree = log2_ceil_usize(
            num_rows
                .next_power_of_two()
                .max(min_height.next_power_of_two()),
        );
        Some((CircuitTableAir::Dynamic(wrapper), degree))
    }
}

/// Returns a type-erased Poseidon2 preprocessor for use when `Val<SC>` is BabyBear, Goldilocks, or KoalaBear.
pub fn poseidon2_preprocessor<F>() -> Box<dyn NpoPreprocessor<F>>
where
    F: StarkField + PrimeField64,
    Poseidon2Preprocessor: NpoPreprocessor<F>,
{
    Box::new(Poseidon2Preprocessor)
}
