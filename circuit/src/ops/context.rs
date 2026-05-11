use alloc::boxed::Box;

use hashbrown::HashMap;
use p3_field::PrimeCharacteristicRing;

use super::executor::OpExecutionState;
use super::npo::{NpoConfig, NpoPrivateData, NpoTypeId, OpStateMap};
use crate::CircuitError;
use crate::types::{NonPrimitiveOpId, WitnessId};

/// Execution context providing operations access to witness table, private data, and configs
///
/// This context is passed to operation executors to give them access to all necessary
/// runtime state without exposing internal implementation details.
pub struct ExecutionContext<'a, F> {
    /// Mutable reference to witness table for reading/writing values
    witness: &'a mut [Option<F>],
    /// Private data map for non-primitive operations
    non_primitive_op_private_data: &'a [Option<NpoPrivateData>],
    /// Operation configurations
    enabled_ops: &'a HashMap<NpoTypeId, NpoConfig>,
    /// Current operation's NonPrimitiveOpId for error reporting
    operation_id: NonPrimitiveOpId,
    /// Operation-specific execution state storage.
    /// Each operation type can store its own state (e.g., chaining state, row records).
    op_states: &'a mut OpStateMap,
}

impl<'a, F: PrimeCharacteristicRing + Eq> ExecutionContext<'a, F> {
    /// Create a new execution context
    pub const fn new(
        witness: &'a mut [Option<F>],
        non_primitive_op_private_data: &'a [Option<NpoPrivateData>],
        enabled_ops: &'a HashMap<NpoTypeId, NpoConfig>,
        operation_id: NonPrimitiveOpId,
        op_states: &'a mut OpStateMap,
    ) -> Self {
        Self {
            witness,
            non_primitive_op_private_data,
            enabled_ops,
            operation_id,
            op_states,
        }
    }

    /// Get witness value at the given index
    #[inline]
    pub fn get_witness(&self, widx: WitnessId) -> Result<F, CircuitError> {
        let idx = widx.0 as usize;

        #[cfg(debug_assertions)]
        {
            self.witness
                .get(idx)
                .and_then(Option::as_ref)
                .map(p3_field::Dup::dup)
                .ok_or(CircuitError::WitnessNotSet { witness_id: widx })
        }

        #[cfg(not(debug_assertions))]
        unsafe {
            Ok(self
                .witness
                .get_unchecked(idx)
                .as_ref()
                .unwrap_unchecked()
                .dup())
        }
    }

    /// Set witness value at the given index
    #[inline]
    pub fn set_witness(&mut self, widx: WitnessId, value: F) -> Result<(), CircuitError> {
        let idx = widx.0 as usize;

        #[cfg(debug_assertions)]
        {
            let slot = self
                .witness
                .get_mut(idx)
                .ok_or(CircuitError::WitnessIdOutOfBounds { witness_id: widx })?;

            if let Some(existing_value) = slot {
                if *existing_value != value {
                    return Err(CircuitError::WitnessConflict {
                        witness_id: widx,
                        existing: alloc::format!("{existing_value:?}"),
                        new: alloc::format!("{value:?}"),
                        expr_ids: alloc::vec![],
                    });
                }
                return Ok(());
            }

            *slot = Some(value);
        }

        #[cfg(not(debug_assertions))]
        unsafe {
            *self.witness.get_unchecked_mut(idx) = Some(value);
        }

        Ok(())
    }

    /// Get private data for the current operation
    pub fn get_private_data(&self) -> Result<&NpoPrivateData, CircuitError> {
        self.non_primitive_op_private_data
            .get(self.operation_id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(CircuitError::NonPrimitiveOpMissingPrivateData {
                operation_index: self.operation_id,
            })
    }

    /// Get operation configuration by type
    pub fn get_config(&self, op_type: &NpoTypeId) -> Result<&NpoConfig, CircuitError> {
        self.enabled_ops.get(op_type).ok_or_else(|| {
            CircuitError::InvalidNonPrimitiveOpConfiguration {
                op: op_type.clone(),
            }
        })
    }

    /// Get the current operation ID
    #[inline]
    pub const fn operation_id(&self) -> NonPrimitiveOpId {
        self.operation_id
    }

    /// Get operation-specific state for reading.
    ///
    /// Returns `None` if no state has been initialized for this operation type.
    pub fn get_op_state<T: OpExecutionState + 'static>(&self, op_type: &NpoTypeId) -> Option<&T> {
        self.op_states
            .get(op_type)
            .and_then(|state| state.downcast_ref())
    }

    /// Get operation-specific state for mutation, creating default if not present.
    ///
    /// This is the primary way executors should access their state.
    pub fn get_op_state_mut<T: OpExecutionState + Default + 'static>(
        &mut self,
        op_type: &NpoTypeId,
    ) -> &mut T {
        self.op_states
            .entry(op_type.clone())
            .or_insert_with(|| Box::new(T::default()))
            .downcast_mut::<T>()
            .expect("type mismatch in op state - this is a bug")
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::ops::poseidon2_perm::{Poseidon2Config, Poseidon2PermPrivateData};

    type F = BabyBear;

    #[test]
    fn test_execution_context_get_witness() {
        let val = F::from_u64(42);
        let mut witness = vec![Some(val), Some(F::from_u64(100))];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.get_witness(WitnessId(0));
        assert_eq!(result.unwrap(), val);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_execution_context_get_witness_unset() {
        let mut witness = vec![None, Some(F::from_u64(100))];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.get_witness(WitnessId(0));
        assert!(result.is_err());
        match result {
            Err(CircuitError::WitnessNotSet { witness_id }) => {
                assert_eq!(witness_id, WitnessId(0));
            }
            _ => panic!("Expected WitnessNotSet error"),
        }
    }

    #[test]
    fn test_execution_context_set_witness() {
        let mut witness = vec![None, None];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let mut ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let val = F::from_u64(99);
        let result = ctx.set_witness(WitnessId(0), val);
        assert!(result.is_ok());
        assert_eq!(witness[0], Some(val));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_execution_context_set_witness_conflict() {
        let existing_val = F::from_u64(50);
        let mut witness = vec![Some(existing_val)];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let mut ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let new_val = F::from_u64(99);
        let result = ctx.set_witness(WitnessId(0), new_val);
        assert!(result.is_err());
        match result {
            Err(CircuitError::WitnessConflict { witness_id, .. }) => {
                assert_eq!(witness_id, WitnessId(0));
            }
            _ => panic!("Expected WitnessConflict error"),
        }
    }

    #[test]
    fn test_execution_context_set_witness_idempotent() {
        let val = F::from_u64(50);
        let mut witness = vec![Some(val)];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let mut ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.set_witness(WitnessId(0), val);
        assert!(result.is_ok());
        assert_eq!(witness[0], Some(val));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_execution_context_set_witness_out_of_bounds() {
        let mut witness = vec![None];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let mut ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.set_witness(WitnessId(10), F::from_u64(1));
        assert!(result.is_err());
        match result {
            Err(CircuitError::WitnessIdOutOfBounds { witness_id }) => {
                assert_eq!(witness_id, WitnessId(10));
            }
            _ => panic!("Expected WitnessIdOutOfBounds error"),
        }
    }

    #[test]
    fn test_execution_context_get_private_data() {
        let poseidon2_data: Poseidon2PermPrivateData<F> = Poseidon2PermPrivateData {
            sibling: vec![F::ZERO, F::ZERO],
        };
        let private_data = vec![Some(NpoPrivateData::new(poseidon2_data.clone()))];

        let mut witness: Vec<Option<F>> = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();
        let ctx: ExecutionContext<'_, F> =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.get_private_data();
        assert!(result.is_ok());
        let downcast = result
            .unwrap()
            .downcast_ref::<Poseidon2PermPrivateData<F>>()
            .unwrap();
        assert_eq!(*downcast, poseidon2_data);
    }

    #[test]
    fn test_execution_context_get_private_data_missing() {
        let private_data = vec![];
        let mut witness = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let ctx: ExecutionContext<'_, F> =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.get_private_data();
        assert!(result.is_err());
        match result {
            Err(CircuitError::NonPrimitiveOpMissingPrivateData { operation_index }) => {
                assert_eq!(operation_index, op_id);
            }
            _ => panic!("Expected NonPrimitiveOpMissingPrivateData error"),
        }
    }

    #[test]
    fn test_execution_context_get_config() {
        let mut configs = HashMap::new();
        let op_type = NpoTypeId::poseidon2_perm(Poseidon2Config::BabyBearD4Width16);
        configs.insert(op_type.clone(), NpoConfig::new(42u32));

        let mut witness = vec![];
        let private_data = vec![];
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();
        let ctx: ExecutionContext<'_, F> =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let result = ctx.get_config(&op_type);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execution_context_get_config_missing() {
        let configs = HashMap::new();
        let mut witness = vec![];
        let private_data = vec![];
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let ctx: ExecutionContext<'_, F> =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let op_type = NpoTypeId::poseidon2_perm(Poseidon2Config::BabyBearD4Width16);
        let result = ctx.get_config(&op_type);

        assert!(result.is_err());
        match result {
            Err(CircuitError::InvalidNonPrimitiveOpConfiguration { .. }) => {}
            _ => panic!("Expected InvalidNonPrimitiveOpConfiguration error"),
        }
    }

    #[test]
    fn test_execution_context_operation_id() {
        let mut witness = vec![];
        let private_data = vec![];
        let configs = HashMap::new();
        let expected_id = NonPrimitiveOpId(42);
        let mut op_states = BTreeMap::new();
        let ctx: ExecutionContext<'_, F> = ExecutionContext::new(
            &mut witness,
            &private_data,
            &configs,
            expected_id,
            &mut op_states,
        );

        let retrieved_id = ctx.operation_id();
        assert_eq!(retrieved_id, expected_id);
    }

    /// Test state type for verifying generic state management.
    #[derive(Debug, Default)]
    struct TestOpState {
        value: Option<u64>,
    }

    #[test]
    fn test_execution_context_op_state() {
        let mut witness: Vec<Option<F>> = vec![];
        let private_data = vec![];
        let configs = HashMap::new();
        let op_id = NonPrimitiveOpId(0);
        let mut op_states = BTreeMap::new();

        let mut ctx =
            ExecutionContext::new(&mut witness, &private_data, &configs, op_id, &mut op_states);

        let key = NpoTypeId::poseidon2_perm(Poseidon2Config::BabyBearD4Width16);

        assert!(ctx.get_op_state::<TestOpState>(&key).is_none());

        let state = ctx.get_op_state_mut::<TestOpState>(&key);
        assert!(state.value.is_none());
        state.value = Some(42);

        let state_ref = ctx.get_op_state::<TestOpState>(&key).unwrap();
        assert_eq!(state_ref.value, Some(42));
    }
}
