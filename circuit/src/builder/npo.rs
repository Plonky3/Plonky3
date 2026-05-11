use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use core::marker::PhantomData;

use hashbrown::HashMap;
use p3_field::Field;

use crate::CircuitBuilderError;
use crate::ops::{HintExecutor, NpoConfig, NpoTypeId, Op};
use crate::tables::TraceGeneratorFn;
use crate::types::{ExprId, NonPrimitiveOpId, WitnessId};

/// Per-op extra parameters that are not encoded in the op type.
#[derive(Debug)]
pub enum NonPrimitiveOpParams<F> {
    Poseidon2Perm { new_start: bool, merkle_path: bool },
    Unconstrained { executor: Box<dyn HintExecutor<F>> },
    Recompose,
}

impl<F> NonPrimitiveOpParams<F> {
    /// Return the `(new_start, merkle_path)` flags if this is a Poseidon2 permutation.
    pub const fn as_poseidon2_perm(&self) -> Option<(bool, bool)> {
        match self {
            Self::Poseidon2Perm {
                new_start,
                merkle_path,
            } => Some((*new_start, *merkle_path)),
            _ => None,
        }
    }

    /// Returns `true` if this is the `Recompose` variant.
    pub const fn is_recompose(&self) -> bool {
        matches!(self, Self::Recompose)
    }
}

impl<F: Field> Clone for NonPrimitiveOpParams<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Poseidon2Perm {
                new_start,
                merkle_path,
            } => Self::Poseidon2Perm {
                new_start: *new_start,
                merkle_path: *merkle_path,
            },
            Self::Unconstrained { executor } => Self::Unconstrained {
                executor: executor.boxed(),
            },
            Self::Recompose => Self::Recompose,
        }
    }
}

/// The non-primitive operation id, type, the vectors of the expressions representing its inputs
/// and outputs, and any per-op parameters.
#[derive(Debug, Clone)]
pub struct NonPrimitiveOperationData<F: Field> {
    pub op_id: NonPrimitiveOpId,
    pub op_type: NpoTypeId,
    /// Input expressions (e.g., for Poseidon2Perm: [in0, in1, in2, in3, mmcs_index_sum, mmcs_bit])
    pub input_exprs: Vec<Vec<ExprId>>,
    /// Output expressions (e.g., for Poseidon2Perm: [out0, out1])
    pub output_exprs: Vec<Vec<ExprId>>,
    pub params: Option<NonPrimitiveOpParams<F>>,
}

/// Lowering context passed to `NpoCircuitPlugin::lower`, providing access to the
/// expression-to-witness map and witness allocation function.
pub struct NpoLoweringContext<'a, F> {
    pub expr_to_widx: &'a mut HashMap<ExprId, WitnessId>,
    pub alloc_witness_id: &'a mut dyn FnMut(usize) -> WitnessId,
    /// Phantom to keep `F` in the type, even though we only carry witness IDs here.
    _phantom: PhantomData<F>,
}

impl<'a, F> NpoLoweringContext<'a, F> {
    pub fn new(
        expr_to_widx: &'a mut HashMap<ExprId, WitnessId>,
        alloc_witness_id: &'a mut dyn FnMut(usize) -> WitnessId,
    ) -> Self {
        Self {
            expr_to_widx,
            alloc_witness_id,
            _phantom: PhantomData,
        }
    }

    /// Look up the witness index for a given expression.
    ///
    /// # Returns
    ///
    /// The previously-assigned witness index.
    ///
    /// # Errors
    ///
    /// Returns `MissingExprMapping` if the expression was never assigned a witness slot.
    pub fn resolve_witness_id(
        &self,
        expr_id: ExprId,
        context: &str,
    ) -> Result<WitnessId, CircuitBuilderError> {
        self.expr_to_widx.get(&expr_id).copied().ok_or_else(|| {
            CircuitBuilderError::MissingExprMapping {
                expr_id,
                context: context.into(),
            }
        })
    }

    /// Return the witness index for an expression, allocating a fresh one if absent.
    pub fn ensure_witness_id(&mut self, expr_id: ExprId) -> WitnessId {
        *self
            .expr_to_widx
            .entry(expr_id)
            .or_insert_with(|| (self.alloc_witness_id)(expr_id.0 as usize))
    }

    /// Convert expression slots to witness slots, preserving the slot structure.
    ///
    /// Each input slot must contain at most one expression.
    /// Empty slots produce empty witness vectors; single-element slots are resolved
    /// through the expression-to-witness map.
    ///
    /// # Errors
    ///
    /// - Returns `NonPrimitiveOpArity` if any slot has more than one element.
    /// - Returns `MissingExprMapping` if a single-element slot has no witness mapping.
    pub fn lower_expr_slots(
        &self,
        slots: &[Vec<ExprId>],
        op_name: &'static str,
        context_prefix: &str,
    ) -> Result<Vec<Vec<WitnessId>>, CircuitBuilderError> {
        let mut result = Vec::with_capacity(slots.len());
        for (i, slot) in slots.iter().enumerate() {
            if slot.len() > 1 {
                return Err(CircuitBuilderError::NonPrimitiveOpArity {
                    op: op_name,
                    expected: format!("0 or 1 element per {context_prefix}"),
                    got: slot.len(),
                });
            }
            let witness_index = slot
                .iter()
                .map(|&expr| {
                    self.resolve_witness_id(expr, &format!("{op_name} {context_prefix} {i}"))
                })
                .collect::<Result<_, _>>()?;
            result.push(witness_index);
        }
        Ok(result)
    }
}

/// Circuit-layer plugin interface for non-primitive operations.
///
/// Implementors are responsible for:
/// - Lowering their high-level operation description into a single `Op<F>`
/// - Providing a trace generator for their dedicated table
/// - Exposing their configuration as an `NpoConfig`
pub trait NpoCircuitPlugin<F: Field>: Send + Sync {
    /// Unique type identifier for this NPO (e.g. "poseidon2_perm/baby_bear_d4_w16").
    fn type_id(&self) -> NpoTypeId;

    /// Convert a high-level NPO operation into a concrete `Op<F>`.
    ///
    /// The lowering context gives access to the expression→witness mapping and
    /// witness allocation for any new outputs.
    fn lower(
        &self,
        data: &NonPrimitiveOperationData<F>,
        output_exprs: &[(u32, ExprId)],
        ctx: &mut NpoLoweringContext<'_, F>,
    ) -> Result<Op<F>, CircuitBuilderError>;

    /// Produce the trace generator for this NPO.
    fn trace_generator(&self) -> TraceGeneratorFn<F>;

    /// Return plugin-specific configuration for this NPO.
    fn config(&self) -> NpoConfig;
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hashbrown::HashMap;
    use p3_test_utils::baby_bear_params::BabyBear;

    use super::*;

    type F = BabyBear;

    #[test]
    fn as_poseidon2_perm_returns_flags() {
        let params: NonPrimitiveOpParams<F> = NonPrimitiveOpParams::Poseidon2Perm {
            new_start: true,
            merkle_path: false,
        };
        assert_eq!(params.as_poseidon2_perm(), Some((true, false)));
    }

    #[test]
    fn as_poseidon2_perm_returns_none_for_recompose() {
        let params: NonPrimitiveOpParams<F> = NonPrimitiveOpParams::Recompose;
        assert_eq!(params.as_poseidon2_perm(), None);
    }

    #[test]
    fn is_recompose_true() {
        let params: NonPrimitiveOpParams<F> = NonPrimitiveOpParams::Recompose;
        assert!(params.is_recompose());
    }

    #[test]
    fn is_recompose_false_for_poseidon2() {
        let params: NonPrimitiveOpParams<F> = NonPrimitiveOpParams::Poseidon2Perm {
            new_start: false,
            merkle_path: false,
        };
        assert!(!params.is_recompose());
    }

    #[test]
    fn resolve_witness_id_found() {
        let mut map = HashMap::new();
        map.insert(ExprId(5), WitnessId(42));
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        assert_eq!(
            ctx.resolve_witness_id(ExprId(5), "test").unwrap(),
            WitnessId(42)
        );
    }

    #[test]
    fn resolve_witness_id_missing_errors() {
        let mut map = HashMap::new();
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let Err(CircuitBuilderError::MissingExprMapping { expr_id, context }) =
            ctx.resolve_witness_id(ExprId(99), "some context")
        else {
            panic!("expected MissingExprMapping");
        };
        assert_eq!(expr_id, ExprId(99));
        assert_eq!(context, "some context");
    }

    #[test]
    fn ensure_witness_id_existing() {
        let mut map = HashMap::new();
        map.insert(ExprId(3), WitnessId(10));
        let mut alloc = |_: usize| WitnessId(999);
        let mut ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        assert_eq!(ctx.ensure_witness_id(ExprId(3)), WitnessId(10));
    }

    #[test]
    fn ensure_witness_id_allocates_new() {
        let mut map = HashMap::new();
        let mut counter = 100u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let mut ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let wid = ctx.ensure_witness_id(ExprId(7));
        assert_eq!(wid, WitnessId(100));
        // Subsequent call returns the same id.
        assert_eq!(ctx.ensure_witness_id(ExprId(7)), WitnessId(100));
    }

    #[test]
    fn lower_expr_slots_maps_populated_slots() {
        let mut map = HashMap::new();
        map.insert(ExprId(1), WitnessId(10));
        map.insert(ExprId(2), WitnessId(20));
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let slots = vec![vec![ExprId(1)], vec![], vec![ExprId(2)]];
        let result = ctx.lower_expr_slots(&slots, "TestOp", "input").unwrap();

        assert_eq!(
            result,
            vec![vec![WitnessId(10)], vec![], vec![WitnessId(20)]]
        );
    }

    #[test]
    fn lower_expr_slots_rejects_multi_element_slot() {
        let mut map = HashMap::new();
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let slots = vec![vec![ExprId(1), ExprId(2)]];
        let Err(CircuitBuilderError::NonPrimitiveOpArity { op, expected, got }) =
            ctx.lower_expr_slots(&slots, "TestOp", "input")
        else {
            panic!("expected NonPrimitiveOpArity");
        };
        assert_eq!(op, "TestOp");
        assert_eq!(expected, "0 or 1 element per input");
        assert_eq!(got, 2);
    }

    #[test]
    fn lower_expr_slots_errors_on_unmapped_expr() {
        let mut map = HashMap::new();
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let slots = vec![vec![ExprId(5)]];
        let Err(CircuitBuilderError::MissingExprMapping { expr_id, context }) =
            ctx.lower_expr_slots(&slots, "TestOp", "input")
        else {
            panic!("expected MissingExprMapping");
        };
        assert_eq!(expr_id, ExprId(5));
        assert_eq!(context, "TestOp input 0");
    }

    #[test]
    fn lower_expr_slots_empty_input() {
        let mut map = HashMap::new();
        let mut alloc = |_: usize| WitnessId(999);
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let slots: Vec<Vec<ExprId>> = vec![];
        let result = ctx.lower_expr_slots(&slots, "TestOp", "input").unwrap();

        assert!(result.is_empty());
    }
}
