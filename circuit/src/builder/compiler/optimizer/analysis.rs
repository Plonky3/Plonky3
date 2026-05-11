use crate::ops::AluOpKind;
use crate::types::WitnessId;

/// Lightweight summary of how a witness was produced.
///
/// Only encodes the shapes that MulAdd fusion needs to inspect
/// (`Const`, `Mul`); every other operation is collapsed into `Other`
/// because the fusion passes do not need to reason about them.
#[derive(Clone, Debug)]
pub(super) enum OpDef<F> {
    Const(F),
    Mul { a: WitnessId, b: WitnessId },
    Other,
}

impl<F> OpDef<F> {
    pub(super) const fn is_const(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    pub(super) const fn as_mul(&self) -> Option<(WitnessId, WitnessId)> {
        match self {
            Self::Mul { a, b } => Some((*a, *b)),
            _ => None,
        }
    }
}

/// An operation definition paired with its position in the op list.
#[derive(Clone, Debug)]
pub(super) struct IndexedDef<F> {
    pub(super) idx: usize,
    pub(super) def: OpDef<F>,
}

impl<F> IndexedDef<F> {
    pub(super) const fn new(idx: usize, def: OpDef<F>) -> Self {
        Self { idx, def }
    }
}

/// Dedup key for ALU operations, normalizing commutative operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct AluKey {
    kind: AluOpKind,
    a: u32,
    b: u32,
    c: u32,
}

impl AluKey {
    /// Builds a dedup key, sorting operands for commutative ops.
    pub(super) fn new(kind: AluOpKind, a: WitnessId, b: WitnessId, c: Option<WitnessId>) -> Self {
        match kind {
            AluOpKind::Add | AluOpKind::Mul => Self {
                kind,
                a: a.0.min(b.0),
                b: a.0.max(b.0),
                c: 0,
            },
            AluOpKind::BoolCheck => Self {
                kind,
                a: a.0,
                b: b.0,
                c: 0,
            },
            AluOpKind::MulAdd | AluOpKind::HornerAcc => Self {
                kind,
                a: a.0,
                b: b.0,
                c: c.unwrap_or(WitnessId(0)).0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    #[test]
    fn opdef_is_const() {
        assert!(OpDef::<F>::Const(F::ZERO).is_const());
        assert!(!OpDef::<F>::Other.is_const());
        assert!(
            !OpDef::<F>::Mul {
                a: WitnessId(0),
                b: WitnessId(1)
            }
            .is_const()
        );
    }

    #[test]
    fn opdef_as_mul() {
        let def = OpDef::<F>::Mul {
            a: WitnessId(3),
            b: WitnessId(7),
        };
        assert_eq!(def.as_mul(), Some((WitnessId(3), WitnessId(7))));

        assert_eq!(OpDef::<F>::Other.as_mul(), None);
        assert_eq!(OpDef::<F>::Const(F::ONE).as_mul(), None);
    }

    #[test]
    fn indexed_def_fields() {
        let def = IndexedDef::new(42, OpDef::<F>::Other);
        assert_eq!(def.idx, 42);
        assert!(matches!(def.def, OpDef::Other));
    }

    #[test]
    fn alu_key_commutative_add() {
        let k1 = AluKey::new(AluOpKind::Add, WitnessId(5), WitnessId(3), None);
        let k2 = AluKey::new(AluOpKind::Add, WitnessId(3), WitnessId(5), None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn alu_key_commutative_mul() {
        let k1 = AluKey::new(AluOpKind::Mul, WitnessId(10), WitnessId(2), None);
        let k2 = AluKey::new(AluOpKind::Mul, WitnessId(2), WitnessId(10), None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn alu_key_non_commutative_muladd() {
        let k1 = AluKey::new(
            AluOpKind::MulAdd,
            WitnessId(1),
            WitnessId(2),
            Some(WitnessId(3)),
        );
        let k2 = AluKey::new(
            AluOpKind::MulAdd,
            WitnessId(2),
            WitnessId(1),
            Some(WitnessId(3)),
        );
        assert_ne!(k1, k2);
    }

    #[test]
    fn alu_key_different_kinds_differ() {
        let k_add = AluKey::new(AluOpKind::Add, WitnessId(1), WitnessId(2), None);
        let k_mul = AluKey::new(AluOpKind::Mul, WitnessId(1), WitnessId(2), None);
        assert_ne!(k_add, k_mul);
    }

    #[test]
    fn alu_key_boolcheck_preserves_order() {
        let k1 = AluKey::new(AluOpKind::BoolCheck, WitnessId(5), WitnessId(3), None);
        let k2 = AluKey::new(AluOpKind::BoolCheck, WitnessId(3), WitnessId(5), None);
        // BoolCheck doesn't normalize order
        assert_ne!(k1, k2);
    }

    #[test]
    fn alu_key_muladd_c_none_defaults_to_zero() {
        let k1 = AluKey::new(AluOpKind::MulAdd, WitnessId(1), WitnessId(2), None);
        let k2 = AluKey::new(
            AluOpKind::MulAdd,
            WitnessId(1),
            WitnessId(2),
            Some(WitnessId(0)),
        );
        assert_eq!(k1, k2);
    }

    proptest! {
        #[test]
        fn alu_key_add_commutative_prop(a in 0u32..1000, b in 0u32..1000) {
            let k1 = AluKey::new(AluOpKind::Add, WitnessId(a), WitnessId(b), None);
            let k2 = AluKey::new(AluOpKind::Add, WitnessId(b), WitnessId(a), None);
            prop_assert_eq!(k1, k2);
        }

        #[test]
        fn alu_key_mul_commutative_prop(a in 0u32..1000, b in 0u32..1000) {
            let k1 = AluKey::new(AluOpKind::Mul, WitnessId(a), WitnessId(b), None);
            let k2 = AluKey::new(AluOpKind::Mul, WitnessId(b), WitnessId(a), None);
            prop_assert_eq!(k1, k2);
        }
    }
}
