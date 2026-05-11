use alloc::vec::Vec;

use crate::types::WitnessId;

/// Witness value store.
///
/// Holds all intermediate computation values produced during circuit execution.
/// It exists only as a convenient value store for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessTrace<F> {
    /// Sequential witness IDs: WitnessId(0), WitnessId(1), WitnessId(2), ...
    ///
    /// Kept for API compatibility with existing tests and tools.
    pub index: Vec<WitnessId>,

    /// Witness field element values.
    values: Vec<F>,
}

impl<F> WitnessTrace<F> {
    /// Create a new instance from a flat vector of values.
    ///
    /// IDs are assigned sequentially starting from `WitnessId(0)`.
    pub fn new(values: Vec<F>) -> Self {
        let mut index = Vec::with_capacity(values.len());
        for i in 0..values.len() as u32 {
            index.push(WitnessId(i));
        }
        Self { index, values }
    }

    /// Number of witness entries.
    pub const fn num_rows(&self) -> usize {
        self.values.len()
    }

    /// Return a reference to the value at the given witness ID.
    /// Returns `None` if `witness_id` is out of bounds.
    pub fn get_value(&self, witness_id: WitnessId) -> Option<&F> {
        self.values.get(witness_id.0 as usize)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_new_single_value() {
        let trace = WitnessTrace::new(vec![F::from_u64(42)]);

        assert_eq!(
            trace,
            WitnessTrace {
                index: vec![WitnessId(0)],
                values: vec![F::from_u64(42)],
            }
        );
    }

    #[test]
    fn test_new_multiple_values() {
        let vals = vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)];
        let trace = WitnessTrace::new(vals.clone());

        assert_eq!(
            trace,
            WitnessTrace {
                index: vec![WitnessId(0), WitnessId(1), WitnessId(2)],
                values: vals,
            }
        );
    }

    #[test]
    fn test_new_empty() {
        let trace = WitnessTrace::<F>::new(vec![]);

        assert_eq!(
            trace,
            WitnessTrace {
                index: vec![],
                values: vec![],
            }
        );
        assert_eq!(trace.num_rows(), 0);
    }

    #[test]
    fn test_num_rows() {
        let trace = WitnessTrace::new(vec![F::ONE, F::TWO, F::ZERO]);
        assert_eq!(trace.num_rows(), 3);
    }

    #[test]
    fn test_get_value_valid() {
        let trace = WitnessTrace::new(vec![F::from_u64(5), F::from_u64(10), F::from_u64(15)]);

        assert_eq!(trace.get_value(WitnessId(0)), Some(&F::from_u64(5)));
        assert_eq!(trace.get_value(WitnessId(1)), Some(&F::from_u64(10)));
        assert_eq!(trace.get_value(WitnessId(2)), Some(&F::from_u64(15)));
    }

    #[test]
    fn test_get_value_out_of_bounds() {
        let trace = WitnessTrace::new(vec![F::ONE]);

        assert_eq!(trace.get_value(WitnessId(1)), None);
        assert_eq!(trace.get_value(WitnessId(100)), None);
    }

    #[test]
    fn test_sequential_ids() {
        let n = 5;
        let vals: Vec<F> = (0..n).map(|i| F::from_u64(i as u64)).collect();
        let trace = WitnessTrace::new(vals);

        for i in 0..n {
            assert_eq!(trace.index[i], WitnessId(i as u32));
        }
    }
}
