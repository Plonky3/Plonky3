//! Poseidon2 trace types and trace generation.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::any::Any;

use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField};

use crate::CircuitError;
use crate::ops::NpoTypeId;
use crate::ops::poseidon2_perm::config::Poseidon2Config;
use crate::ops::poseidon2_perm::state::Poseidon2ExecutionState;
use crate::tables::NonPrimitiveTrace;
use crate::types::NonPrimitiveOpId;

/// Trait to provide Poseidon2 configuration parameters for a field type.
///
/// This allows the trace generator and AIR to work with different Poseidon2 configurations
/// without hardcoding parameters. Implementations should provide the standard
/// parameters for their field type.
pub trait Poseidon2Params {
    type BaseField: PrimeField + PrimeCharacteristicRing;
    /// Poseidon2 configuration key for this parameter set.
    const CONFIG: Poseidon2Config;
    /// Extension degree D
    const D: usize = Self::CONFIG.d();
    /// Total width in base field elements
    const WIDTH: usize = Self::CONFIG.width();

    /// Rate in extension elements
    const RATE_EXT: usize = Self::CONFIG.rate_ext();
    /// Capacity in extension elements
    const CAPACITY_EXT: usize = Self::CONFIG.capacity_ext();
    /// Capacity size in base field elements = CAPACITY_EXT * D
    const CAPACITY_SIZE: usize = Self::CAPACITY_EXT * Self::D;

    /// S-box degree (polynomial degree for the S-box)
    const SBOX_DEGREE: u64 = Self::CONFIG.sbox_degree();
    /// Number of S-box registers
    const SBOX_REGISTERS: usize = Self::CONFIG.sbox_registers();

    /// Number of half full rounds
    const HALF_FULL_ROUNDS: usize = Self::CONFIG.half_full_rounds();
    /// Number of partial rounds
    const PARTIAL_ROUNDS: usize = Self::CONFIG.partial_rounds();

    /// Width in extension elements = RATE_EXT + CAPACITY_EXT
    const WIDTH_EXT: usize = Self::RATE_EXT + Self::CAPACITY_EXT;
}

/// BabyBear D=1 Width=16 configuration for base field challenges.
///
/// This is used when the challenge type is the base field itself (no extension).
/// The Poseidon2 permutation operates directly on 16 base field elements.
pub struct BabyBearD1Width16;

impl Poseidon2Params for BabyBearD1Width16 {
    type BaseField = p3_baby_bear::BabyBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::BabyBearD1Width16;
}

/// KoalaBear D=1 Width=16 configuration for base field challenges.
///
/// This is used when the challenge type is the base field itself (no extension).
/// The Poseidon2 permutation operates directly on 16 base field elements.
pub struct KoalaBearD1Width16;

impl Poseidon2Params for KoalaBearD1Width16 {
    type BaseField = p3_koala_bear::KoalaBear;
    const CONFIG: Poseidon2Config = Poseidon2Config::KoalaBearD1Width16;
}

/// Goldilocks D=2 Width=8 configuration (matches Poseidon2Goldilocks<8>).
pub struct GoldilocksD2Width8;

impl Poseidon2Params for GoldilocksD2Width8 {
    type BaseField = p3_goldilocks::Goldilocks;
    const CONFIG: Poseidon2Config = Poseidon2Config::GoldilocksD2Width8;
}

/// Poseidon2 operation table row.
///
/// This implements the Poseidon Permutation Table specification.
/// See: https://github.com/Plonky3/Plonky3-recursion/discussions/186
///
/// The table has one row per Poseidon call, implementing:
/// - Standard chaining (Challenger-style sponge use)
/// - Merkle-path chaining (MMCS directional hashing)
/// - Selective limb exposure to the witness via CTL
/// - Optional MMCS index accumulator
#[derive(Debug, Clone)]
pub struct Poseidon2CircuitRow<F> {
    /// Control: If 1, row begins a new independent Poseidon chain.
    pub new_start: bool,
    /// Control: 0 → normal sponge/Challenger mode, 1 → Merkle-path mode.
    pub merkle_path: bool,
    /// Control: Direction bit for Merkle left/right hashing (only meaningful when merkle_path = 1).
    pub mmcs_bit: bool,
    /// Value: Optional MMCS accumulator (base field, encodes a u32-like integer).
    pub mmcs_index_sum: F,
    /// Inputs to the Poseidon2 permutation (flattened state).
    /// For execution rows: 4 extension limbs. For trace rows: WIDTH base field elements.
    pub input_values: Vec<F>,
    /// Input exposure flags for CTL lookups: permuted to match the physical trace layout.
    pub in_ctl: Vec<bool>,
    /// Input exposure indices for CTL lookups.
    pub input_indices: Vec<u32>,
    /// Output exposure flags for rate limbs (CTL-verified when true).
    pub out_ctl: Vec<bool>,
    /// Output exposure indices: index into the witness table for rate limbs.
    pub output_indices: Vec<u32>,
    /// MMCS index exposure: index for CTL exposure of mmcs_index_sum.
    pub mmcs_index_sum_idx: u32,
    /// Whether mmcs_index_sum CTL is enabled. When false, the mmcs_index_sum lookup is disabled.
    pub mmcs_ctl_enabled: bool,
}

/// Poseidon2 trace for all hash operations in the circuit.
#[derive(Debug, Clone)]
pub struct Poseidon2Trace<F> {
    /// Operation type for this Poseidon2 trace.
    pub op_type: NpoTypeId,
    /// All Poseidon2 operations (permutation rows) in this trace.
    pub operations: Vec<Poseidon2CircuitRow<F>>,
}

impl<F> Poseidon2Trace<F> {
    pub const fn total_rows(&self) -> usize {
        self.operations.len()
    }
}

impl<TraceF: Clone + Send + Sync + 'static, CF> NonPrimitiveTrace<CF> for Poseidon2Trace<TraceF> {
    fn op_type(&self) -> NpoTypeId {
        self.op_type.clone()
    }

    fn rows(&self) -> usize {
        self.total_rows()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn boxed_clone(&self) -> Box<dyn NonPrimitiveTrace<CF>> {
        Box::new(self.clone())
    }
}

/// Generate the Poseidon2 trace from execution state.
///
/// Converts circuit rows from extension field format (4 limbs) to base field format (16 elements).
///
/// # Type Parameters
/// - `F`: The circuit field type (extension field)
/// - `Config`: A type implementing `Poseidon2Params` that specifies the Poseidon2 configuration
pub fn generate_poseidon2_trace<
    F: Field + ExtensionField<Config::BaseField>,
    Config: Poseidon2Params,
>(
    op_states: &crate::ops::OpStateMap,
) -> Result<Option<Box<dyn NonPrimitiveTrace<F>>>, CircuitError> {
    let op_type = NpoTypeId::poseidon2_perm(Config::CONFIG);
    let Some(state) = op_states
        .get(&op_type)
        .and_then(|s| s.downcast_ref::<Poseidon2ExecutionState<F>>())
    else {
        return Ok(None);
    };

    if state.rows.is_empty() {
        return Ok(None);
    }

    let d = Config::D;

    // Convert extension field rows to base field rows
    let operations: Vec<Poseidon2CircuitRow<Config::BaseField>> = state
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| -> Result<_, CircuitError> {
            let limb_count = Config::WIDTH / d;
            // Flatten extension limbs to WIDTH base field elements.
            assert_eq!(
                row.input_values.len(),
                limb_count,
                "Source row must have WIDTH/D input limbs"
            );
            let mut input_values = vec![Config::BaseField::ZERO; Config::WIDTH];
            assert_eq!(
                input_values.len(),
                Config::WIDTH,
                "Target row must have WIDTH input elements"
            );
            for (limb, ext_val) in row.input_values.iter().enumerate() {
                let coeffs = ext_val.as_basis_coefficients_slice();
                if d == 1 {
                    // D=1 AIR consumes one base element per state slot. When the circuit field is an
                    // extension of `BaseField`, embedded-base semantics use the constant coefficient.
                    input_values[limb] = coeffs[0];
                } else {
                    input_values[limb * d..(limb + 1) * d].copy_from_slice(coeffs);
                }
            }

            let mmcs_index_sum = row.mmcs_index_sum.as_base().ok_or_else(|| {
                CircuitError::IncorrectNonPrimitiveOpPrivateData {
                    op: op_type.clone(),
                    operation_index: NonPrimitiveOpId(row_index as u32),
                    expected: "base field mmcs_index_sum".to_string(),
                    got: "extension value".to_string(),
                }
            })?;

            Ok(Poseidon2CircuitRow {
                new_start: row.new_start,
                merkle_path: row.merkle_path,
                mmcs_bit: row.mmcs_bit,
                mmcs_index_sum,
                input_values,
                in_ctl: row.in_ctl.clone(),
                input_indices: row.input_indices.clone(),
                out_ctl: row.out_ctl.clone(),
                output_indices: row.output_indices.clone(),
                mmcs_index_sum_idx: row.mmcs_index_sum_idx,
                mmcs_ctl_enabled: row.mmcs_ctl_enabled,
            })
        })
        .collect::<Result<Vec<_>, CircuitError>>()?;

    Ok(Some(Box::new(Poseidon2Trace {
        op_type,
        operations,
    })))
}
