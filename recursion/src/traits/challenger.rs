//! Trait for recursive Fiat-Shamir challenger operations within circuits.
//!
//! This module provides the [`RecursiveChallenger`] trait which mirrors the native
//! Plonky3 `DuplexChallenger` API. The trait supports both base field and extension
//! field operations to ensure exact transcript compatibility.

use alloc::vec::Vec;

use p3_circuit::{CircuitBuilder, CircuitBuilderError};
use p3_field::{ExtensionField, PrimeField64};

use crate::Target;

/// Trait for performing Fiat-Shamir transformations within a circuit.
///
/// This trait mirrors the native Plonky3 `DuplexChallenger` API:
/// - `observe` / `sample` operate on base field elements
/// - `observe_ext` / `sample_ext` operate on extension field elements
///
/// The circuit challenger maintains state as coefficient-level targets to ensure
/// exact transcript compatibility with the native challenger, including correct
/// handling of partial absorbs.
///
/// # Type Parameters
/// - `BF`: The base prime field
/// - `EF`: The extension field over `BF`
pub trait RecursiveChallenger<BF: PrimeField64, EF: ExtensionField<BF>> {
    // ========================================================================
    // Base field operations (match native DuplexChallenger's observe/sample)
    // ========================================================================

    /// Observe a base field element in the Fiat-Shamir transcript.
    ///
    /// Matches the native `DuplexChallenger::observe` behavior exactly.
    /// The value is pushed to the input buffer and duplexing occurs when
    /// the buffer reaches RATE elements.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder for creating operations
    /// - `value`: A target representing a base field element (embedded in EF)
    fn observe(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target);

    /// Observe multiple base field elements.
    fn observe_slice(&mut self, circuit: &mut CircuitBuilder<EF>, values: &[Target]) {
        for &value in values {
            self.observe(circuit, value);
        }
    }

    /// Sample a base field element from the sponge.
    ///
    /// Matches the native `DuplexChallenger::sample` behavior exactly.
    /// If there are pending inputs or the output buffer is empty, duplexing
    /// occurs first.
    ///
    /// # Returns
    /// A target representing a base field element (embedded in EF)
    fn sample(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target;

    // ========================================================================
    // Extension field operations (match native observe_algebra_element/sample_algebra_element)
    // ========================================================================

    /// Observe an extension field element in the Fiat-Shamir transcript.
    ///
    /// Matches the native `FieldChallenger::observe_algebra_element` behavior.
    /// Decomposes the extension element to D base coefficients and observes each.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder for creating operations
    /// - `value`: A target representing an extension field element
    fn observe_ext(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target);

    /// Observe multiple extension field elements.
    fn observe_ext_slice(&mut self, circuit: &mut CircuitBuilder<EF>, values: &[Target]) {
        for &value in values {
            self.observe_ext(circuit, value);
        }
    }

    /// Sample an extension field element from the sponge.
    ///
    /// Matches the native `FieldChallenger::sample_algebra_element` behavior.
    /// Samples D base field elements and recomposes them into an extension element.
    ///
    /// # Returns
    /// A target representing an extension field element
    fn sample_ext(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target;

    /// Sample multiple extension field challenges.
    fn sample_ext_vec(&mut self, circuit: &mut CircuitBuilder<EF>, count: usize) -> Vec<Target> {
        (0..count).map(|_| self.sample_ext(circuit)).collect()
    }

    // ========================================================================
    // Bit operations (for PoW and query indices)
    // ========================================================================

    /// Sample bits from a base field element.
    ///
    /// Samples a base field element and decomposes it to bits.
    /// This is used for sampling query indices in FRI.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder for creating operations
    /// - `num_bits`: Number of bits to return
    ///
    /// # Returns
    /// Vector of `num_bits` bits as targets (each in {0, 1})
    fn sample_bits(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        num_bits: usize,
    ) -> Result<Vec<Target>, CircuitBuilderError>;

    /// Verify a proof-of-work witness.
    ///
    /// Observes the witness as a base field element, samples a challenge,
    /// decomposes it to bits, and verifies that the first `witness_bits` bits
    /// are all zero.
    ///
    /// # Parameters
    /// - `circuit`: Circuit builder for creating operations
    /// - `witness_bits`: Number of leading bits that must be zero
    /// - `witness`: The proof-of-work witness target (base field element)
    fn check_pow_witness(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        witness_bits: usize,
        witness: Target,
    ) -> Result<(), CircuitBuilderError>;

    /// Clear the challenger state.
    ///
    /// Resets the internal sponge state and buffers.
    fn clear(&mut self, circuit: &mut CircuitBuilder<EF>);
}
