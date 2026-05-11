//! Transcript compatibility tests for CircuitChallenger vs native DuplexChallenger.
//!
//! These tests verify that the recursive CircuitChallenger produces identical
//! transcript values as the native Plonky3 DuplexChallenger.

mod common;

use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger};
use p3_circuit::ops::{Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit::{CircuitBuilder, Traces};
use p3_field::PrimeField64;
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::challenger::CircuitChallenger;
use p3_recursion::traits::RecursiveChallenger;

// ============================================================================
// BabyBear D=4, WIDTH=16, RATE=8
// ============================================================================

mod baby_bear_d4 {
    use p3_test_utils::baby_bear_params::*;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn setup_circuit_with_poseidon2() -> CircuitBuilder<EF> {
        let mut circuit = CircuitBuilder::<EF>::new();
        let perm = default_babybear_poseidon2_16();
        circuit.enable_poseidon2_perm::<BabyBearD4Width16, _>(
            generate_poseidon2_trace::<EF, BabyBearD4Width16>,
            perm,
        );
        circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        circuit
    }

    /// Test basic observe/sample transcript compatibility.
    #[test]
    fn test_transcript_single_observe_sample() {
        let perm = default_babybear_poseidon2_16();

        // Native challenger
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);

        // Circuit challenger
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe a single value
        let val = F::from_u64(42);
        native.observe(val);
        let val_target = circuit.define_const(EF::from(val));
        RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, val_target);

        // Fill to RATE to trigger duplexing
        for i in 1..RATE {
            let v = F::from_u64(i as u64);
            native.observe(v);
            let v_t = circuit.define_const(EF::from(v));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, v_t);
        }

        // Sample and compare
        let native_sample: F = native.sample();
        let circuit_sample =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);

        // Connect circuit sample to expected native value
        let expected = circuit.define_const(EF::from(native_sample));
        circuit.connect(circuit_sample, expected);

        // Build and run - if values match, no WitnessConflict
        let compiled = circuit.build().expect("Circuit should build");
        let runner = compiled.runner();
        let traces: Traces<EF> = runner
            .run()
            .expect("Single observe/sample should match native");

        assert!(
            traces.witness_trace.num_rows() > 0,
            "Should produce witness trace"
        );
    }

    /// Test observe_ext matches native observe_base_as_algebra_element.
    #[test]
    fn test_transcript_observe_ext_compatibility() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe a base value as algebra element (like batch-STARK does)
        let base_val = F::from_usize(123);
        native.observe_base_as_algebra_element::<EF>(base_val);
        let val_target = circuit.define_const(EF::from(base_val));
        RecursiveChallenger::<F, EF>::observe_ext(
            &mut circuit_challenger,
            &mut circuit,
            val_target,
        );

        // Observe another value
        let base_val2 = F::from_usize(456);
        native.observe_base_as_algebra_element::<EF>(base_val2);
        let val_target2 = circuit.define_const(EF::from(base_val2));
        RecursiveChallenger::<F, EF>::observe_ext(
            &mut circuit_challenger,
            &mut circuit,
            val_target2,
        );

        // Sample extension element
        let native_ext: EF = native.sample_algebra_element();
        let circuit_ext =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);

        let expected = circuit.define_const(native_ext);
        circuit.connect(circuit_ext, expected);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("observe_ext should match native observe_base_as_algebra_element");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test multiple duplexing rounds maintain transcript compatibility.
    #[test]
    fn test_transcript_multiple_duplexing_rounds() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // First round: observe RATE elements
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 100);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample after first round
        let native_s1: F = native.sample();
        let circuit_s1 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s1 = circuit.define_const(EF::from(native_s1));
        circuit.connect(circuit_s1, expected_s1);

        // Second round: observe more elements
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 200);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample after second round
        let native_s2: F = native.sample();
        let circuit_s2 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s2 = circuit.define_const(EF::from(native_s2));
        circuit.connect(circuit_s2, expected_s2);

        // Third round with extension samples
        for i in 0..4 {
            let val = F::from_u64(i as u64 + 300);
            native.observe_base_as_algebra_element::<EF>(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe_ext(&mut circuit_challenger, &mut circuit, t);
        }

        let native_ext: EF = native.sample_algebra_element();
        let circuit_ext =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);
        let expected_ext = circuit.define_const(native_ext);
        circuit.connect(circuit_ext, expected_ext);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Multiple duplexing rounds should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test partial absorption (less than RATE) then sample.
    #[test]
    fn test_transcript_partial_absorption() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe only 3 elements (less than RATE=8)
        for i in 0..3 {
            let val = F::from_u64(i as u64 + 50);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample triggers duplexing with partial input
        let native_sample: F = native.sample();
        let circuit_sample =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);

        let expected = circuit.define_const(EF::from(native_sample));
        circuit.connect(circuit_sample, expected);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Partial absorption should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test extension field element observation (observe_algebra_element equivalent).
    #[test]
    fn test_transcript_observe_extension_element() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Create extension field elements
        let ext_val = EF::from_basis_coefficients_slice(&[
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ])
        .unwrap();

        // Native: observe_algebra_element decomposes to D coefficients
        native.observe_algebra_element(ext_val);
        // Circuit: observe_ext does the same decomposition
        let ext_target = circuit.define_const(ext_val);
        RecursiveChallenger::<F, EF>::observe_ext(
            &mut circuit_challenger,
            &mut circuit,
            ext_target,
        );

        // Observe more to trigger duplexing
        let ext_val2 = EF::from_basis_coefficients_slice(&[
            F::from_u64(11),
            F::from_u64(21),
            F::from_u64(31),
            F::from_u64(41),
        ])
        .unwrap();
        native.observe_algebra_element(ext_val2);
        let ext_target2 = circuit.define_const(ext_val2);
        RecursiveChallenger::<F, EF>::observe_ext(
            &mut circuit_challenger,
            &mut circuit,
            ext_target2,
        );

        // Sample and compare
        let native_ext: EF = native.sample_algebra_element();
        let circuit_ext =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);

        let expected = circuit.define_const(native_ext);
        circuit.connect(circuit_ext, expected);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Extension element observation should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test mixed observation types (base and extension).
    #[test]
    fn test_transcript_mixed_observations() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Mix of base field observations
        for i in 0..3 {
            let val = F::from_u64(i as u64);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Extension field observation
        let base_as_ext = F::from_usize(999);
        native.observe_base_as_algebra_element::<EF>(base_as_ext);
        let t = circuit.define_const(EF::from(base_as_ext));
        RecursiveChallenger::<F, EF>::observe_ext(&mut circuit_challenger, &mut circuit, t);

        // More base observations
        for i in 0..2 {
            let val = F::from_u64(i as u64 + 100);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample base field element
        let native_base: F = native.sample();
        let circuit_base =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_base = circuit.define_const(EF::from(native_base));
        circuit.connect(circuit_base, expected_base);

        // Sample extension field element
        let native_ext: EF = native.sample_algebra_element();
        let circuit_ext =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);
        let expected_ext = circuit.define_const(native_ext);
        circuit.connect(circuit_ext, expected_ext);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Mixed observations should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test circuit challenger clear functionality.
    /// Native DuplexChallenger doesn't have clear, so we verify circuit clear
    /// produces consistent state (fresh zero state).
    #[test]
    fn test_transcript_clear_produces_fresh_state() {
        let perm = default_babybear_poseidon2_16();

        // Create a fresh native challenger (simulating what clear does)
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);

        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // First, do some observations to dirty the state
        for i in 0..5 {
            let val = F::from_u64(i as u64);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Clear circuit challenger (resets to fresh zero state)
        RecursiveChallenger::<F, EF>::clear(&mut circuit_challenger, &mut circuit);

        // Now both should be in equivalent fresh states
        // Observe same values in both
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1000);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample should match
        let native_sample: F = native.sample();
        let circuit_sample =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);

        let expected = circuit.define_const(EF::from(native_sample));
        circuit.connect(circuit_sample, expected);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Clear should produce fresh state matching new challenger");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Test multiple consecutive samples without intermediate observations.
    #[test]
    fn test_transcript_consecutive_samples() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Initial observations
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 77);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Multiple consecutive samples
        for _ in 0..5 {
            let native_s: F = native.sample();
            let circuit_s =
                RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
            let expected = circuit.define_const(EF::from(native_s));
            circuit.connect(circuit_s, expected);
        }

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Consecutive samples should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    /// Edge case: Exactly RATE observations triggers duplexing, then sample.
    /// Tests the boundary condition when input buffer is exactly full.
    #[test]
    fn test_edge_case_exactly_rate_observations() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe exactly RATE elements (should trigger duplexing on last observe)
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 500);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // At this point, input buffer should be empty (duplexing occurred)
        // and output buffer should be full

        // Sample should come from output buffer without triggering new duplexing
        let native_s1: F = native.sample();
        let circuit_s1 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s1 = circuit.define_const(EF::from(native_s1));
        circuit.connect(circuit_s1, expected_s1);

        // Sample again to verify output buffer state
        let native_s2: F = native.sample();
        let circuit_s2 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s2 = circuit.define_const(EF::from(native_s2));
        circuit.connect(circuit_s2, expected_s2);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Exactly RATE observations should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Edge case: Drain entire output buffer (RATE samples) then sample again.
    /// This triggers a new duplexing when output buffer is empty.
    #[test]
    fn test_edge_case_drain_output_buffer_completely() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe RATE elements to trigger duplexing
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 600);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Drain entire output buffer (RATE samples)
        for j in 0..RATE {
            let native_s: F = native.sample();
            let circuit_s =
                RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
            let expected = circuit.define_const(EF::from(native_s));
            circuit.connect(circuit_s, expected);

            // Verify we got a valid sample at each step
            if j == RATE - 1 {
                // Last sample from output buffer
            }
        }

        // Now output buffer is empty - this sample should trigger new duplexing
        let native_extra: F = native.sample();
        let circuit_extra =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_extra = circuit.define_const(EF::from(native_extra));
        circuit.connect(circuit_extra, expected_extra);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Draining output buffer then sampling should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Edge case: Interleaved observe/sample pattern.
    /// Tests complex state transitions with alternating operations.
    #[test]
    fn test_edge_case_interleaved_observe_sample() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Pattern: observe a few, sample, observe more, sample, etc.
        // This tests output buffer invalidation on observe

        // Observe 3
        for i in 0..3 {
            let val = F::from_u64(i as u64 + 700);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample (triggers duplexing with 3 inputs)
        let native_s1: F = native.sample();
        let circuit_s1 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s1 = circuit.define_const(EF::from(native_s1));
        circuit.connect(circuit_s1, expected_s1);

        // Observe 2 more (invalidates output buffer)
        for i in 0..2 {
            let val = F::from_u64(i as u64 + 800);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample (triggers new duplexing with 2 inputs)
        let native_s2: F = native.sample();
        let circuit_s2 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s2 = circuit.define_const(EF::from(native_s2));
        circuit.connect(circuit_s2, expected_s2);

        // Sample again (from output buffer)
        let native_s3: F = native.sample();
        let circuit_s3 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s3 = circuit.define_const(EF::from(native_s3));
        circuit.connect(circuit_s3, expected_s3);

        // Observe 1 more
        let val = F::from_u64(900);
        native.observe(val);
        let t = circuit.define_const(EF::from(val));
        RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);

        // Final sample
        let native_s4: F = native.sample();
        let circuit_s4 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s4 = circuit.define_const(EF::from(native_s4));
        circuit.connect(circuit_s4, expected_s4);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Interleaved observe/sample should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Edge case: Sample immediately without any observations.
    /// Tests initial state sampling behavior.
    #[test]
    fn test_edge_case_sample_without_observations() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Sample immediately (duplexing with zero-initialized state)
        let native_s1: F = native.sample();
        let circuit_s1 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s1 = circuit.define_const(EF::from(native_s1));
        circuit.connect(circuit_s1, expected_s1);

        // Sample again
        let native_s2: F = native.sample();
        let circuit_s2 =
            RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected_s2 = circuit.define_const(EF::from(native_s2));
        circuit.connect(circuit_s2, expected_s2);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Sample without observations should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    /// Edge case: Observe single element then sample multiple times.
    /// Tests output buffer usage after minimal input.
    #[test]
    fn test_edge_case_single_observe_multiple_samples() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Single observation
        let val = F::from_u64(12345);
        native.observe(val);
        let t = circuit.define_const(EF::from(val));
        RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);

        // Multiple samples (first triggers duplexing, rest from buffer)
        for _ in 0..RATE {
            let native_s: F = native.sample();
            let circuit_s =
                RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
            let expected = circuit.define_const(EF::from(native_s));
            circuit.connect(circuit_s, expected);
        }

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Single observe then multiple samples should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }

    // ============================================================================
    // sample_bits / PoW grinding tests
    // ============================================================================

    /// Test that `sample_bits(n)` in the circuit produces the same low-n bits as
    /// the native `CanSampleBits::sample_bits(n)` after an identical observation
    /// sequence.
    ///
    /// The native challenger returns a `usize`; bit `k` of that integer must equal
    /// the `k`-th target produced by the circuit challenger.
    #[test]
    fn test_sample_bits_matches_native() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Sync transcripts: observe RATE values in both native and circuit challenger.
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 42);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample the same base field element natively and in the circuit.
        let num_bits = 5usize;
        let native_index: usize = native.sample_bits(num_bits);
        let circuit_bits: Vec<_> = RecursiveChallenger::<F, EF>::sample_bits(
            &mut circuit_challenger,
            &mut circuit,
            num_bits,
        )
        .expect("sample_bits should succeed");

        // Connect each circuit bit to its expected native value.
        assert_eq!(circuit_bits.len(), num_bits);
        for (k, &bit_target) in circuit_bits.iter().enumerate() {
            let expected_bit = ((native_index >> k) & 1) as u64;
            let expected = circuit.define_const(EF::from(F::from_u64(expected_bit)));
            circuit.connect(bit_target, expected);
        }

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("sample_bits circuit bits should match native");
    }

    /// Test `sample_bits` across several calls with different bit counts,
    /// interleaved with new observations, to exercise the full duplex cycle.
    #[test]
    fn test_sample_bits_multiple_calls_match_native() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // First batch: observe + sample_bits(3)
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 10);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        let n1 = 3usize;
        let native_idx1: usize = native.sample_bits(n1);
        let circuit_bits1 =
            RecursiveChallenger::<F, EF>::sample_bits(&mut circuit_challenger, &mut circuit, n1)
                .expect("sample_bits should succeed");
        for (k, &bit) in circuit_bits1.iter().enumerate() {
            let expected =
                circuit.define_const(EF::from(F::from_u64(((native_idx1 >> k) & 1) as u64)));
            circuit.connect(bit, expected);
        }

        // Second batch: observe more + sample_bits(7)
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 200);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        let n2 = 7usize;
        let native_idx2: usize = native.sample_bits(n2);
        let circuit_bits2 =
            RecursiveChallenger::<F, EF>::sample_bits(&mut circuit_challenger, &mut circuit, n2)
                .expect("sample_bits should succeed");
        for (k, &bit) in circuit_bits2.iter().enumerate() {
            let expected =
                circuit.define_const(EF::from(F::from_u64(((native_idx2 >> k) & 1) as u64)));
            circuit.connect(bit, expected);
        }

        // Third call with no new observations: sample_bits(1)
        let n3 = 1usize;
        let native_idx3: usize = native.sample_bits(n3);
        let circuit_bits3 =
            RecursiveChallenger::<F, EF>::sample_bits(&mut circuit_challenger, &mut circuit, n3)
                .expect("sample_bits should succeed");
        let expected3 = circuit.define_const(EF::from(F::from_u64(native_idx3 as u64)));
        circuit.connect(circuit_bits3[0], expected3);

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("multiple sample_bits calls should match native");
    }

    /// Test that `sample_bits` is consistent with `sample`: the low `n` bits of
    /// the sampled field element returned by `sample` must equal the bits returned
    /// by `sample_bits(n)` for an identical transcript.
    ///
    /// We verify this on the native side (where both operations can be compared
    /// directly) and confirm the bit-level agreement using the circuit challenger.
    #[test]
    fn test_sample_bits_consistent_with_sample() {
        let perm = default_babybear_poseidon2_16();
        let num_bits = 4usize;

        // Native: confirm sample+mask equals sample_bits.
        let mut native_a = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm.clone());
        let mut native_b = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 77);
            native_a.observe(val);
            native_b.observe(val);
        }
        let raw_sample: F = native_a.sample();
        let native_mask = (raw_sample.as_canonical_u64() & ((1u64 << num_bits) - 1)) as usize;
        let native_bits: usize = native_b.sample_bits(num_bits);
        assert_eq!(
            native_mask, native_bits,
            "native sample+mask and sample_bits must agree"
        );

        // Circuit: use a single circuit and sample_bits; connect each bit to the
        // expected native bit value derived from native_bits.
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 77);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        let circuit_bits = RecursiveChallenger::<F, EF>::sample_bits(
            &mut circuit_challenger,
            &mut circuit,
            num_bits,
        )
        .expect("sample_bits should succeed");
        assert_eq!(circuit_bits.len(), num_bits);
        for (k, &bit_target) in circuit_bits.iter().enumerate() {
            let expected_val = EF::from(F::from_u64(((native_bits >> k) & 1) as u64));
            let expected = circuit.define_const(expected_val);
            circuit.connect(bit_target, expected);
        }

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("sample_bits should be consistent with sample+mask");
    }

    /// Test that `check_pow_witness` in the circuit correctly verifies a valid
    /// proof-of-work witness.
    ///
    /// We grind for a witness by brute-force on a clone of the native challenger,
    /// then replay the same observation + `check_pow_witness` in the circuit.
    /// A valid witness causes all leading bits to be zero, so `assert_zero` in
    /// the circuit succeeds.
    #[test]
    fn test_check_pow_witness_valid() {
        let perm = default_babybear_poseidon2_16();
        let pow_bits = 2usize; // small so brute-force terminates quickly

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Sync transcripts.
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 55);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Grind: find a base field element w such that, after observing w, the next
        // `pow_bits` sampled bits are all zero.  Clone the native challenger so we
        // can probe multiple candidates without advancing the real transcript.
        let witness = (0u64..)
            .find(|&w| {
                let mut probe = native.clone();
                probe.observe(F::from_u64(w));
                probe.sample_bits(pow_bits) == 0
            })
            .expect("brute-force PoW witness should terminate");
        let witness_f = F::from_u64(witness);

        // Verify natively: observe witness, check pow_bits leading bits are zero.
        native.observe(witness_f);
        let native_check = native.sample_bits(pow_bits) == 0;
        assert!(native_check, "native PoW witness check must pass");

        // Verify in circuit.
        let witness_target = circuit.define_const(EF::from(witness_f));
        RecursiveChallenger::<F, EF>::check_pow_witness(
            &mut circuit_challenger,
            &mut circuit,
            pow_bits,
            witness_target,
        )
        .expect("check_pow_witness should succeed");

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("valid PoW witness should satisfy circuit constraints");
    }

    /// Test that `check_pow_witness` with `pow_bits = 0` is a no-op: the circuit
    /// state is unchanged and running succeeds without any additional Poseidon2
    /// calls.
    #[test]
    fn test_check_pow_witness_zero_bits_is_noop() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // With 0 bits, check_pow_witness is a no-op; the witness value is irrelevant.
        let dummy_witness = circuit.define_const(EF::from(F::from_u64(999)));
        RecursiveChallenger::<F, EF>::check_pow_witness(
            &mut circuit_challenger,
            &mut circuit,
            0,
            dummy_witness,
        )
        .expect("check_pow_witness(0) should succeed");

        // The transcript should still agree with native (no witness was observed).
        let native_s: F = native.sample();
        let circuit_s = RecursiveChallenger::<F, EF>::sample(&mut circuit_challenger, &mut circuit);
        let expected = circuit.define_const(EF::from(native_s));
        circuit.connect(circuit_s, expected);

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("zero-bit PoW check should leave transcript unchanged");
    }

    /// Edge case: Extension field samples draining output buffer.
    /// Each sample_ext consumes D base elements from output.
    #[test]
    fn test_edge_case_extension_samples_drain_buffer() {
        let perm = default_babybear_poseidon2_16();

        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit_with_poseidon2();
        let mut circuit_challenger =
            CircuitChallenger::<WIDTH, RATE, Poseidon2Config>::new_babybear();

        // Observe RATE elements
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1000);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut circuit_challenger, &mut circuit, t);
        }

        // Sample extension elements (each consumes 4 base elements from RATE=8 buffer)
        // After 2 ext samples, buffer is empty
        let native_ext1: EF = native.sample_algebra_element();
        let circuit_ext1 =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);
        let expected_ext1 = circuit.define_const(native_ext1);
        circuit.connect(circuit_ext1, expected_ext1);

        let native_ext2: EF = native.sample_algebra_element();
        let circuit_ext2 =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);
        let expected_ext2 = circuit.define_const(native_ext2);
        circuit.connect(circuit_ext2, expected_ext2);

        // Third ext sample should trigger new duplexing
        let native_ext3: EF = native.sample_algebra_element();
        let circuit_ext3 =
            RecursiveChallenger::<F, EF>::sample_ext(&mut circuit_challenger, &mut circuit);
        let expected_ext3 = circuit.define_const(native_ext3);
        circuit.connect(circuit_ext3, expected_ext3);

        let compiled = circuit.build().expect("Circuit should build");
        let traces: Traces<EF> = compiled
            .runner()
            .run()
            .expect("Extension samples draining buffer should match native");

        assert!(traces.witness_trace.num_rows() > 0);
    }
}

// ============================================================================
// KoalaBear D=4, WIDTH=16, RATE=8
// ============================================================================

mod koala_bear_d4 {
    use p3_poseidon2_circuit_air::KoalaBearD4Width16;
    use p3_test_utils::koala_bear_params::*;

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    fn setup_circuit() -> CircuitBuilder<EF> {
        let mut circuit = CircuitBuilder::<EF>::new();
        let perm = default_koalabear_poseidon2_16();
        circuit.enable_poseidon2_perm::<KoalaBearD4Width16, _>(
            generate_poseidon2_trace::<EF, KoalaBearD4Width16>,
            perm,
        );
        circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        circuit
    }

    const fn new_challenger() -> CircuitChallenger<WIDTH, RATE, Poseidon2Config> {
        CircuitChallenger::new_koalabear()
    }

    /// Basic observe/sample transcript compatibility.
    #[test]
    fn test_koalabear_d4_observe_sample() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let native_s: F = native.sample();
        let circuit_s = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
        let expected = circuit.define_const(EF::from(native_s));
        circuit.connect(circuit_s, expected);

        let compiled = circuit.build().expect("circuit should build");
        compiled
            .runner()
            .run()
            .expect("KoalaBear D4 observe/sample should match native");
    }

    /// Multiple duplexing rounds stay in sync.
    #[test]
    fn test_koalabear_d4_multiple_rounds() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for round in 0..3u64 {
            for i in 0..RATE {
                let val = F::from_u64(round * 100 + i as u64);
                native.observe(val);
                let t = circuit.define_const(EF::from(val));
                RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
            }
            let ns: F = native.sample();
            let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
            let exp = circuit.define_const(EF::from(ns));
            circuit.connect(cs, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D4 multiple rounds should match native");
    }

    /// Extension field observe/sample compatibility.
    #[test]
    fn test_koalabear_d4_ext_observe_sample() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        // observe_base_as_algebra_element decomposes a base element as D coefficients.
        for i in 0..4u64 {
            let val = F::from_u64(i + 10);
            native.observe_base_as_algebra_element::<EF>(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe_ext(&mut cc, &mut circuit, t);
        }

        let ne: EF = native.sample_algebra_element();
        let ce = RecursiveChallenger::<F, EF>::sample_ext(&mut cc, &mut circuit);
        let exp = circuit.define_const(ne);
        circuit.connect(ce, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D4 ext observe/sample should match native");
    }

    /// `sample_bits` output matches the low-n bits of the natively sampled field element.
    #[test]
    fn test_koalabear_d4_sample_bits() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 50);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let num_bits = 5usize;
        let native_idx: usize = native.sample_bits(num_bits);
        let bits = RecursiveChallenger::<F, EF>::sample_bits(&mut cc, &mut circuit, num_bits)
            .expect("sample_bits should succeed");

        assert_eq!(bits.len(), num_bits);
        for (k, &bit) in bits.iter().enumerate() {
            let exp = circuit.define_const(EF::from(F::from_u64(((native_idx >> k) & 1) as u64)));
            circuit.connect(bit, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D4 sample_bits should match native");
    }

    /// `check_pow_witness` verifies a valid PoW witness (brute-forced with a small bit count).
    #[test]
    fn test_koalabear_d4_check_pow_witness() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 77);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let pow_bits = 2usize;
        let witness = (0u64..)
            .find(|&w| {
                let mut probe = native.clone();
                probe.observe(F::from_u64(w));
                probe.sample_bits(pow_bits) == 0
            })
            .expect("brute-force PoW witness should terminate");
        let witness_f = F::from_u64(witness);

        native.observe(witness_f);
        assert!(native.sample_bits(pow_bits) == 0, "native PoW must pass");

        let wt = circuit.define_const(EF::from(witness_f));
        RecursiveChallenger::<F, EF>::check_pow_witness(&mut cc, &mut circuit, pow_bits, wt)
            .expect("check_pow_witness should succeed");

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D4 valid PoW witness should satisfy circuit");
    }

    /// Partial absorption (< RATE observations) then sample.
    #[test]
    fn test_koalabear_d4_partial_absorption() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..3 {
            let val = F::from_u64(i as u64 + 200);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let ns: F = native.sample();
        let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
        let exp = circuit.define_const(EF::from(ns));
        circuit.connect(cs, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D4 partial absorption should match native");
    }
}

// ============================================================================
// KoalaBear D=1, WIDTH=16, RATE=8  (base field challenges)
// ============================================================================

mod koala_bear_d1 {
    // For D=1 the base field IS the extension field.
    use p3_challenger::DuplexChallenger;
    use p3_circuit::ops::KoalaBearD1Width16;
    use p3_test_utils::koala_bear_params::*;

    use super::*;

    type F = KoalaBear;

    // For D=1, CircuitBuilder runs over F directly.
    fn setup_circuit() -> CircuitBuilder<F> {
        let mut circuit = CircuitBuilder::<F>::new();
        let perm = default_koalabear_poseidon2_16();
        circuit.enable_poseidon2_perm_base::<KoalaBearD1Width16, _>(
            generate_poseidon2_trace::<F, KoalaBearD1Width16>,
            perm,
        );
        circuit.enable_recompose::<F>(generate_recompose_trace::<F, F>);
        circuit
    }

    const fn new_challenger() -> CircuitChallenger<WIDTH, RATE, Poseidon2Config> {
        CircuitChallenger::new_koalabear_base()
    }

    /// Basic observe/sample transcript compatibility for D=1 base field challenger.
    #[test]
    fn test_koalabear_d1_observe_sample() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1);
            native.observe(val);
            let t = circuit.define_const(val);
            RecursiveChallenger::<F, F>::observe(&mut cc, &mut circuit, t);
        }

        let native_s: F = native.sample();
        let circuit_s = RecursiveChallenger::<F, F>::sample(&mut cc, &mut circuit);
        let expected = circuit.define_const(native_s);
        circuit.connect(circuit_s, expected);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D1 observe/sample should match native");
    }

    /// Multiple duplexing rounds.
    #[test]
    fn test_koalabear_d1_multiple_rounds() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for round in 0..3u64 {
            for i in 0..RATE {
                let val = F::from_u64(round * 100 + i as u64 + 1);
                native.observe(val);
                let t = circuit.define_const(val);
                RecursiveChallenger::<F, F>::observe(&mut cc, &mut circuit, t);
            }
            let ns: F = native.sample();
            let cs = RecursiveChallenger::<F, F>::sample(&mut cc, &mut circuit);
            let exp = circuit.define_const(ns);
            circuit.connect(cs, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D1 multiple rounds should match native");
    }

    /// `sample_bits` output matches the low-n bits of the natively sampled field element.
    #[test]
    fn test_koalabear_d1_sample_bits() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 33);
            native.observe(val);
            let t = circuit.define_const(val);
            RecursiveChallenger::<F, F>::observe(&mut cc, &mut circuit, t);
        }

        let num_bits = 4usize;
        let native_idx: usize = native.sample_bits(num_bits);
        let bits = RecursiveChallenger::<F, F>::sample_bits(&mut cc, &mut circuit, num_bits)
            .expect("sample_bits should succeed");

        assert_eq!(bits.len(), num_bits);
        for (k, &bit) in bits.iter().enumerate() {
            let exp = circuit.define_const(F::from_u64(((native_idx >> k) & 1) as u64));
            circuit.connect(bit, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D1 sample_bits should match native");
    }

    /// Partial absorption then sample.
    #[test]
    fn test_koalabear_d1_partial_absorption() {
        let perm = default_koalabear_poseidon2_16();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..3 {
            let val = F::from_u64(i as u64 + 500);
            native.observe(val);
            let t = circuit.define_const(val);
            RecursiveChallenger::<F, F>::observe(&mut cc, &mut circuit, t);
        }

        let ns: F = native.sample();
        let cs = RecursiveChallenger::<F, F>::sample(&mut cc, &mut circuit);
        let exp = circuit.define_const(ns);
        circuit.connect(cs, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("KoalaBear D1 partial absorption should match native");
    }
}

// ============================================================================
// Goldilocks D=2, WIDTH=8, RATE=4
// ============================================================================

mod goldilocks_d2 {
    use p3_circuit::ops::GoldilocksD2Width8;
    use p3_test_utils::goldilocks_params::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    fn make_perm() -> Perm {
        let mut rng = SmallRng::seed_from_u64(1);
        Perm::new_from_rng_128(&mut rng)
    }

    fn setup_circuit() -> CircuitBuilder<EF> {
        let mut circuit = CircuitBuilder::<EF>::new();
        circuit.enable_poseidon2_perm_width_8::<GoldilocksD2Width8, _>(
            generate_poseidon2_trace::<EF, GoldilocksD2Width8>,
            make_perm(),
        );
        circuit.enable_recompose::<F>(generate_recompose_trace::<F, EF>);
        circuit
    }

    const fn new_challenger() -> CircuitChallenger<WIDTH, RATE, Poseidon2Config> {
        CircuitChallenger::new_goldilocks()
    }

    /// Basic observe/sample transcript compatibility.
    #[test]
    fn test_goldilocks_d2_observe_sample() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let native_s: F = native.sample();
        let circuit_s = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
        let expected = circuit.define_const(EF::from(native_s));
        circuit.connect(circuit_s, expected);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 observe/sample should match native");
    }

    /// Multiple duplexing rounds.
    #[test]
    fn test_goldilocks_d2_multiple_rounds() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for round in 0..3u64 {
            for i in 0..RATE {
                let val = F::from_u64(round * 100 + i as u64 + 1);
                native.observe(val);
                let t = circuit.define_const(EF::from(val));
                RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
            }
            let ns: F = native.sample();
            let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
            let exp = circuit.define_const(EF::from(ns));
            circuit.connect(cs, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 multiple rounds should match native");
    }

    /// Extension field observe/sample (D=2 extension elements).
    #[test]
    fn test_goldilocks_d2_ext_observe_sample() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        // Use observe_base_as_algebra_element to match RecursiveChallenger::observe_ext.
        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 42);
            native.observe_base_as_algebra_element::<EF>(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe_ext(&mut cc, &mut circuit, t);
        }

        let ne: EF = native.sample_algebra_element();
        let ce = RecursiveChallenger::<F, EF>::sample_ext(&mut cc, &mut circuit);
        let exp = circuit.define_const(ne);
        circuit.connect(ce, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 ext observe/sample should match native");
    }

    /// `sample_bits` output matches native.
    #[test]
    fn test_goldilocks_d2_sample_bits() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 99);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let num_bits = 5usize;
        let native_idx: usize = native.sample_bits(num_bits);
        let bits = RecursiveChallenger::<F, EF>::sample_bits(&mut cc, &mut circuit, num_bits)
            .expect("sample_bits should succeed");

        assert_eq!(bits.len(), num_bits);
        for (k, &bit) in bits.iter().enumerate() {
            let exp = circuit.define_const(EF::from(F::from_u64(((native_idx >> k) & 1) as u64)));
            circuit.connect(bit, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 sample_bits should match native");
    }

    /// Partial absorption (< RATE observations) then sample.
    #[test]
    fn test_goldilocks_d2_partial_absorption() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..2 {
            let val = F::from_u64(i as u64 + 11);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let ns: F = native.sample();
        let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
        let exp = circuit.define_const(EF::from(ns));
        circuit.connect(cs, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 partial absorption should match native");
    }

    /// Consecutive samples drain and refill the output buffer.
    #[test]
    fn test_goldilocks_d2_consecutive_samples() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 300);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        // Sample more than RATE to force re-duplexing.
        for _ in 0..(RATE + 2) {
            let ns: F = native.sample();
            let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
            let exp = circuit.define_const(EF::from(ns));
            circuit.connect(cs, exp);
        }

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 consecutive samples should match native");
    }

    /// `check_pow_witness` with 0 bits is a no-op; transcript remains in sync.
    #[test]
    fn test_goldilocks_d2_pow_zero_bits_noop() {
        let perm = make_perm();
        let mut native = DuplexChallenger::<F, _, WIDTH, RATE>::new(perm);
        let mut circuit = setup_circuit();
        let mut cc = new_challenger();

        for i in 0..RATE {
            let val = F::from_u64(i as u64 + 1);
            native.observe(val);
            let t = circuit.define_const(EF::from(val));
            RecursiveChallenger::<F, EF>::observe(&mut cc, &mut circuit, t);
        }

        let dummy = circuit.define_const(EF::ZERO);
        RecursiveChallenger::<F, EF>::check_pow_witness(&mut cc, &mut circuit, 0, dummy)
            .expect("zero-bit PoW should be a no-op");

        let ns: F = native.sample();
        let cs = RecursiveChallenger::<F, EF>::sample(&mut cc, &mut circuit);
        let exp = circuit.define_const(EF::from(ns));
        circuit.connect(cs, exp);

        circuit
            .build()
            .expect("circuit should build")
            .runner()
            .run()
            .expect("Goldilocks D2 zero-bit PoW should leave transcript unchanged");
    }
}
