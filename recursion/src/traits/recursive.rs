use alloc::vec;
use alloc::vec::Vec;

use p3_circuit::CircuitBuilder;
use p3_field::Field;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::LookupGadget;

use crate::Target;

/// Trait for converting a non-recursive type into its circuit representation.
///
/// Types implementing this trait can be used in recursive verification circuits.
/// The trait handles allocation of circuit targets and extraction of field element values.
pub trait Recursive<F: Field> {
    /// The non-recursive type associated with this recursive type.
    type Input;

    /// Creates a new instance of the recursive type by allocating targets in the circuit.
    ///
    /// This method should allocate all necessary public/private inputs and return
    /// a structure containing the allocated targets.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder to allocate targets in
    /// - `input`: The non-recursive input (used only for structure, not values)
    fn new(circuit: &mut CircuitBuilder<F>, input: &Self::Input) -> Self;

    /// Extracts private field element values from the input.
    ///
    /// Values returned here will be used to populate private inputs during execution.
    /// Default implementation returns an empty vector (no private inputs).
    ///
    /// # Parameters
    /// - `input`: The non-recursive input to extract private values from
    fn get_private_values(_input: &Self::Input) -> Vec<F> {
        vec![]
    }

    /// Extracts public field element values from the input.
    ///
    /// Values returned here will be used to populate public inputs during execution.
    /// The order must match the order in which targets were allocated in `new()`.
    ///
    /// # Parameters
    /// - `input`: The non-recursive input to extract public values from
    fn get_values(input: &Self::Input) -> Vec<F>;
}

pub trait RecursiveLookupGadget<F: Field>: LookupGadget {
    fn verify_global_final_value_circuit(
        &self,
        circuit: &mut CircuitBuilder<F>,
        all_expected_cumulative: &[Target],
    );
}

impl<F: Field> RecursiveLookupGadget<F> for LogUpGadget {
    fn verify_global_final_value_circuit(
        &self,
        circuit: &mut CircuitBuilder<F>,
        all_expected_cumulative: &[Target],
    ) {
        let mut final_cumulative = circuit.define_const(F::ZERO);
        for a_e_c in all_expected_cumulative {
            final_cumulative = circuit.add(final_cumulative, *a_e_c);
        }

        circuit.assert_zero(final_cumulative);
    }
}
