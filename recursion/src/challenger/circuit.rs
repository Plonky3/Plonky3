//! Circuit-based challenger implementation matching native DuplexChallenger exactly.
//!
//! This module provides [`CircuitChallenger`], which maintains state as coefficient-level
//! targets to ensure exact transcript compatibility with the native `DuplexChallenger`.
//!
//! # Soundness
//!
//! All Poseidon2 permutations in the challenger are CTL-verified against the Poseidon2 AIR table.
//! The circuit builder's `add_poseidon2_perm_for_challenger` / `add_poseidon2_perm_for_challenger_base`
//! (in `p3_circuit`) delegate to the standard Poseidon2 non-primitive op with full input and rate-output CTL exposure,
//! and the executor runs the real permutation so the lookup argument enforces correctness.

use alloc::vec;
use alloc::vec::Vec;

use p3_circuit::ops::Poseidon2Config;
use p3_circuit::{CircuitBuilder, CircuitBuilderError};
use p3_field::{ExtensionField, PrimeField64};

use crate::Target;
use crate::challenger_perm::ChallengerPermConfig;
use crate::traits::RecursiveChallenger;

/// Circuit challenger with coefficient-level state management.
///
/// Maintains state as WIDTH base field coefficient targets to exactly match
/// the native `DuplexChallenger<F, P, WIDTH, RATE>` behavior.
///
/// # Type Parameters
/// - `WIDTH`: Sponge state width (16 for Poseidon2)
/// - `RATE`: Sponge rate (8 for typical configuration)
/// - `C`: Challenger permutation config (e.g. [`Poseidon2Config`])
pub struct CircuitChallenger<const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig> {
    /// Permutation config for the challenger (e.g. Poseidon2).
    config: C,

    /// Sponge state: WIDTH base field coefficient targets.
    /// Each target represents a base field element embedded in EF.
    state: Vec<Target>,

    /// Buffered inputs not yet absorbed into state.
    input_buffer: Vec<Target>,

    /// Buffered outputs from last duplexing.
    output_buffer: Vec<Target>,

    /// Whether the challenger has been initialized with zero state.
    initialized: bool,

    /// Whether `duplexing_base` has been called at least once since the last init/clear.
    ///
    /// The first D=1 permutation uses `new_start=true` with rate slots CTL-verified and
    /// capacity `None` (zeros enforced by the compact D=1 AIR). Later permutations use
    /// `new_start=false` with the same capacity pattern and chaining.
    duplexed_once: bool,
}

impl<const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig>
    CircuitChallenger<WIDTH, RATE, C>
{
    /// Create a new uninitialized circuit challenger.
    ///
    /// # Parameters
    /// - `config`: The permutation configuration (e.g. Poseidon2) for the challenger.
    ///
    /// Call `init()` to initialize the state with zeros before use.
    pub const fn new(config: C) -> Self {
        Self {
            config,
            state: Vec::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            initialized: false,
            duplexed_once: false,
        }
    }

    /// Initialize the challenger state with zeros.
    ///
    /// This must be called before any observe/sample operations.
    pub fn init<BF, EF>(&mut self, circuit: &mut CircuitBuilder<EF>)
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        if self.initialized {
            return;
        }
        let zero = circuit.define_const(EF::ZERO);
        self.state = vec![zero; WIDTH];
        self.initialized = true;
    }

    /// Perform duplexing: absorb inputs, permute, fill output buffer.
    ///
    /// Matches native `DuplexChallenger::duplexing()` exactly.
    fn duplexing<BF, EF>(&mut self, circuit: &mut CircuitBuilder<EF>)
    where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        debug_assert!(self.initialized, "Challenger must be initialized");
        debug_assert!(self.input_buffer.len() <= RATE, "Input buffer exceeds RATE");

        let poseidon2_config = *self
            .config
            .as_poseidon2()
            .expect("only Poseidon2 challenger permutation is supported");

        // 1. Overwrite state[0..n] with inputs (NOT XOR, matches native)
        for (i, val) in self.input_buffer.drain(..).enumerate() {
            self.state[i] = val;
        }

        // Branch by Poseidon2 NPO packing (`config.d()`), not `EF::DIMENSION`, so a
        // quintic (or other) challenge field can still use base width-16 Poseidon2.
        if poseidon2_config.d() == 1 {
            self.duplexing_base(circuit, poseidon2_config);
        } else {
            self.duplexing_ext::<BF, EF>(circuit, poseidon2_config);
        }

        // 5. Fill output buffer from state[0..RATE]
        self.output_buffer.clear();
        self.output_buffer.extend_from_slice(&self.state[..RATE]);
    }

    /// Duplexing for D=1 (base field): permutation operates directly on 16 elements.
    ///
    /// The first call uses `new_start=true`: rate slots are CTL-verified; capacity slots use
    /// `None` (initial state is zero and the compact D=1 AIR asserts zero capacity on sponge
    /// chain starts). Subsequent calls use `new_start=false` with `None` for capacity so the
    /// AIR enforces continuity via the chain constraint.
    fn duplexing_base<EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon2_config: Poseidon2Config,
    ) where
        EF: p3_field::Field,
    {
        let (new_start, inputs) = if !self.duplexed_once {
            // First permutation: CTL-verify rate only; capacity is zero without witness CTL.
            let inputs: [Option<Target>; 16] =
                core::array::from_fn(|i| if i < RATE { Some(self.state[i]) } else { None });
            (true, inputs)
        } else {
            // Subsequent permutations: CTL-verify rate inputs only; capacity via chain.
            let inputs: [Option<Target>; 16] =
                core::array::from_fn(|i| if i < RATE { Some(self.state[i]) } else { None });
            (false, inputs)
        };
        self.duplexed_once = true;

        let outputs = circuit
            .add_poseidon2_perm_for_challenger_base(poseidon2_config, new_start, inputs)
            .expect("poseidon2 base permutation should succeed");

        self.state = outputs.to_vec();
    }

    fn duplexing_ext<BF, EF>(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        poseidon2_config: Poseidon2Config,
    ) where
        BF: PrimeField64,
        EF: ExtensionField<BF>,
    {
        let num_ext_limbs = WIDTH / EF::DIMENSION;
        let mut ext_inputs = Vec::with_capacity(num_ext_limbs);
        for i in 0..num_ext_limbs {
            let start = i * EF::DIMENSION;
            let end = start + EF::DIMENSION;
            let ext = circuit
                .recompose_base_coeffs_to_ext::<BF>(&self.state[start..end])
                .expect("recomposition should succeed");
            ext_inputs.push(ext);
        }

        let ext_outputs = circuit
            .add_poseidon2_perm_for_challenger(poseidon2_config, &ext_inputs)
            .expect("poseidon2 permutation should succeed");

        for (limb, &ext_out) in ext_outputs.iter().enumerate() {
            let coeffs = circuit
                .decompose_ext_to_base_coeffs::<BF>(ext_out)
                .expect("decomposition should succeed");
            let start = limb * EF::DIMENSION;
            for (i, coeff) in coeffs.into_iter().enumerate() {
                self.state[start + i] = coeff;
            }
        }
    }
}

impl<const WIDTH: usize, const RATE: usize> CircuitChallenger<WIDTH, RATE, Poseidon2Config> {
    /// Create a challenger with BabyBear D4 Width16 configuration (default).
    pub const fn new_babybear() -> Self {
        Self::new(Poseidon2Config::BabyBearD4Width16)
    }

    /// Create a challenger with BabyBear D1 Width16 configuration (base field challenges).
    pub const fn new_babybear_base() -> Self {
        Self::new(Poseidon2Config::BabyBearD1Width16)
    }

    /// Create a challenger with KoalaBear D4 Width16 configuration.
    pub const fn new_koalabear() -> Self {
        Self::new(Poseidon2Config::KoalaBearD4Width16)
    }

    /// Create a challenger with KoalaBear D1 Width16 configuration (base field challenges).
    pub const fn new_koalabear_base() -> Self {
        Self::new(Poseidon2Config::KoalaBearD1Width16)
    }
}

impl CircuitChallenger<8, 4, Poseidon2Config> {
    /// Create a challenger with Goldilocks D2 Width8 configuration.
    pub const fn new_goldilocks() -> Self {
        Self::new(Poseidon2Config::GoldilocksD2Width8)
    }
}

impl<BF, EF, const WIDTH: usize, const RATE: usize, C: ChallengerPermConfig>
    RecursiveChallenger<BF, EF> for CircuitChallenger<WIDTH, RATE, C>
where
    BF: PrimeField64,
    EF: ExtensionField<BF>,
{
    fn observe(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target) {
        // Ensure initialized
        self.init::<BF, EF>(circuit);

        // Any buffered output is now invalid (matches native behavior)
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == RATE {
            self.duplexing::<BF, EF>(circuit);
        }
    }

    fn sample(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target {
        // Ensure initialized
        self.init::<BF, EF>(circuit);

        // If we have buffered inputs or ran out of outputs, duplex
        // (matches native DuplexChallenger::sample behavior)
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing::<BF, EF>(circuit);
        }

        self.output_buffer
            .pop()
            .expect("Output buffer should be non-empty after duplexing")
    }

    fn observe_ext(&mut self, circuit: &mut CircuitBuilder<EF>, value: Target) {
        // Decompose extension element to D base coefficients
        let coeffs = circuit
            .decompose_ext_to_base_coeffs::<BF>(value)
            .expect("decomposition should succeed");

        // Observe each coefficient (matches native observe_algebra_element)
        for coeff in coeffs {
            self.observe(circuit, coeff);
        }
    }

    fn sample_ext(&mut self, circuit: &mut CircuitBuilder<EF>) -> Target {
        // Sample D base elements (matches native sample_algebra_element)
        let coeffs: Vec<_> = (0..EF::DIMENSION).map(|_| self.sample(circuit)).collect();

        // Recompose into extension element
        circuit
            .recompose_base_coeffs_to_ext::<BF>(&coeffs)
            .expect("recomposition should succeed")
    }

    fn sample_bits(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        num_bits: usize,
    ) -> Result<Vec<Target>, CircuitBuilderError> {
        let base_sample = self.sample(circuit);
        // Decompose base field element to bits
        // We decompose the full base field bit width to ensure correct reconstruction
        let bits = circuit.decompose_to_bits::<BF>(base_sample, BF::bits())?;
        Ok(bits[..num_bits].to_vec())
    }

    fn check_pow_witness(
        &mut self,
        circuit: &mut CircuitBuilder<EF>,
        witness_bits: usize,
        witness: Target,
    ) -> Result<(), CircuitBuilderError> {
        // When no PoW is required, keep challenger state unchanged
        if witness_bits == 0 {
            return Ok(());
        }

        // Observe witness as base field element
        self.observe(circuit, witness);

        // Sample and check leading bits are zero
        let bits = self.sample_bits(circuit, witness_bits)?;
        for bit in bits {
            circuit.assert_zero(bit);
        }

        Ok(())
    }

    fn clear(&mut self, circuit: &mut CircuitBuilder<EF>) {
        let zero = circuit.define_const(EF::ZERO);
        self.state = vec![zero; WIDTH];
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.initialized = true;
        self.duplexed_once = false;
    }
}
