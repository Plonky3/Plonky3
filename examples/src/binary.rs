//! A multi-STARK proving harness for AIRs over `BinaryField128`, using the binary PCS.
//!
//! [`prove_binary_air`] proves and verifies one AIR instance end to end: it derives the
//! commitment scheme's arity from the trace shape, builds a [`BinaryStarkConfig`], times the
//! prove and verify phases, and reports the composed security bits from
//! [`p3_multi_stark::security_report`]. The PCS is binding but not hiding; proofs built here
//! carry no zero-knowledge guarantee.

use core::fmt;
use std::time::Instant;

use p3_air::{Air, BaseAir};
use p3_binary_field::{BinaryChallenger, BinaryField2, BinaryField128};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams, BinaryPcsProverData,
    GroupedCodewordMmcs,
};
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;
use p3_lookup::InteractionSymbolicBuilder;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multi_stark::config::{MultiStarkConfig, PcsError, PcsProverError};
use p3_multi_stark::folder::{InteractionMultilinearFolder, MultilinearFolder};
use p3_multi_stark::packed_ext::PackedExt;
use p3_multi_stark::subfield::{SubfieldAcc, SubfieldVar};
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, ProvingError, SecurityError, SubfieldBackend,
    VerificationError, VerifierInstance, VerifierInstances, prove_with_backend, security_report,
    setup, verify,
};
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_util::{log2_ceil_usize, log2_strict_usize};

type F = BinaryField128;
type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MerkleMmcs = p3_merkle_tree::MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
type Mmcs = GroupedCodewordMmcs<MerkleMmcs>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Multi-STARK configuration proving AIRs over `BinaryField128` with the binary PCS.
pub struct BinaryStarkConfig {
    pcs: BinaryPcs<Mmcs>,
}

impl MultiStarkConfig for BinaryStarkConfig {
    type Val = F;
    type Challenge = F;
    type Challenger = Challenger;
    type Pcs = BinaryPcs<Mmcs>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        // Keccak-256 is shared by the transcript and Merkle tree.
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        // The binary PCS folds at least one variable and does not pad individual tables.
        1
    }

    fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
        SuffixProver::<F, F>::new_witness(tables, 0)
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a BinaryPcsProverData<Mmcs>,
        table_index: usize,
    ) -> &'a Table<F> {
        prover_data.table(table_index)
    }
}

/// Derives a [`BinaryStarkConfig`] for a stacked polynomial of `arity` variables.
///
/// `folding` batches up to that many sequential variable folds between PCS commitments; it is
/// clamped to `arity`, since a batch cannot fold more variables than the polynomial has.
pub fn binary_config(
    arity: usize,
    params: BinaryPcsParams,
    folding: usize,
) -> Result<BinaryStarkConfig, BinaryPcsConfigError> {
    let pcs_config = BinaryPcsConfig::try_new_with_folding(arity, params, folding.min(arity))?;
    let merkle = MerkleMmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let mmcs = Mmcs::for_folding(merkle, &pcs_config);
    Ok(BinaryStarkConfig {
        pcs: BinaryPcs::new(pcs_config, mmcs),
    })
}

/// A fresh transcript seeded for one commit, prove, or verify call.
fn binary_challenger() -> Challenger {
    Challenger::from_hasher(b"p3-examples-binary-hash-air-v1".to_vec(), Keccak256Hash)
}

/// Tunable parameters for [`prove_binary_air`].
///
/// Defaults match `multi-stark/examples/prove_binary_field.rs`.
#[derive(Clone, Copy, Debug)]
pub struct BinaryProofOptions {
    /// Log of the inverse code rate for the binary PCS.
    pub log_inv_rate: usize,
    /// Grinding bits the binary PCS demands once, before its query phase.
    pub pcs_pow_bits: usize,
    /// Composed security target of the whole proof, in bits.
    ///
    /// The binary PCS caps it at `125 - arity - log_inv_rate` once its queries are sampled.
    pub security_bits: usize,
    /// Sequential variable folds batched between binary-PCS commitments.
    pub folding: usize,
    /// Grinding bits demanded per sumcheck round.
    pub sumcheck_pow_bits: usize,
}

impl Default for BinaryProofOptions {
    fn default() -> Self {
        Self {
            log_inv_rate: 2,
            pcs_pow_bits: 0,
            security_bits: 100,
            folding: 3,
            sumcheck_pow_bits: 0,
        }
    }
}

/// Measurements from one [`prove_binary_air`] run.
#[derive(Clone, Copy, Debug)]
pub struct BinaryProofReport {
    /// Trace row count.
    pub rows: usize,
    /// Trace column count.
    pub width: usize,
    /// Number of variables in the stacked polynomial the PCS commits to.
    pub stacked_variables: usize,
    /// Serialized proof size, in bytes.
    pub proof_bytes: usize,
    /// Wall-clock time to lay the trace out as a table and run `prove`.
    pub prove_seconds: f64,
    /// Wall-clock time spent in `verify`.
    pub verify_seconds: f64,
    /// Composed security bits reported by `p3_multi_stark::security_report`.
    pub security_bits: f64,
}

impl fmt::Display for BinaryProofReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rows: {}", self.rows)?;
        writeln!(f, "Width: {}", self.width)?;
        writeln!(f, "Stacked variables: {}", self.stacked_variables)?;
        writeln!(f, "Proof size: {} bytes", self.proof_bytes)?;
        writeln!(f, "Prove time: {:.3}s", self.prove_seconds)?;
        writeln!(f, "Verify time: {:.3}s", self.verify_seconds)?;
        write!(f, "Composed security: {:.2} bits", self.security_bits)
    }
}

/// Failure constructing the config, setting up keys, proving, or verifying a binary AIR.
#[derive(Debug)]
pub enum BinaryProofError {
    /// The requested PCS parameters do not describe a usable binary-PCS schedule.
    Config(BinaryPcsConfigError),
    /// Proving (including `setup`) rejected its configuration, budget, or security target.
    Prove(ProvingError<PcsProverError<BinaryStarkConfig>>),
    /// The generated proof failed verification.
    Verify(VerificationError<PcsError<BinaryStarkConfig>>),
    /// The statement's security assessment left a component unassessed or below target.
    Security(SecurityError),
}

impl From<BinaryPcsConfigError> for BinaryProofError {
    fn from(error: BinaryPcsConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<ProvingError<PcsProverError<BinaryStarkConfig>>> for BinaryProofError {
    fn from(error: ProvingError<PcsProverError<BinaryStarkConfig>>) -> Self {
        Self::Prove(error)
    }
}

impl From<VerificationError<PcsError<BinaryStarkConfig>>> for BinaryProofError {
    fn from(error: VerificationError<PcsError<BinaryStarkConfig>>) -> Self {
        Self::Verify(error)
    }
}

/// AIR obligations the binary-field harness needs, stated once each.
///
/// `BinaryField128` is its own packing and its own extension packing (`F::Packing = F` and
/// `EF::ExtensionPacking = F`), so in [`p3_multi_stark::folder::ProverAir`]'s bound list the
/// scalar-base, packed-base, and verifier instantiations of each folder all become
/// `<'a, F, F, F>`. Naming that trait directly as `ProverAir<F, F>` leaves the compiler unable to
/// choose among the resulting duplicate supertrait obligations; this trait states each distinct
/// one exactly once, so its blanket impl below is what callers actually need to satisfy.
///
/// The last bound is the folder [`SubfieldBackend`] evaluates the first zerocheck round with,
/// inside `GF(4)`.
pub trait BinaryAir:
    BaseAir<F>
    + Air<InteractionSymbolicBuilder<F, F>>
    + for<'a> Air<MultilinearFolder<'a, F, F, F>>
    + for<'a> Air<MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, F, F>>
    + for<'a> Air<InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
    + for<'a> Air<
        MultilinearFolder<'a, F, SubfieldVar<F, BinaryField2>, SubfieldAcc<F, BinaryField2>>,
    >
{
}

impl<A> BinaryAir for A where
    A: BaseAir<F>
        + Air<InteractionSymbolicBuilder<F, F>>
        + for<'a> Air<MultilinearFolder<'a, F, F, F>>
        + for<'a> Air<MultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, F, F>>
        + for<'a> Air<InteractionMultilinearFolder<'a, F, PackedExt<F, F>, PackedExt<F, F>>>
        + for<'a> Air<
            MultilinearFolder<'a, F, SubfieldVar<F, BinaryField2>, SubfieldAcc<F, BinaryField2>>,
        >
{
}

/// Proves and verifies `air` against `trace`, reporting size and timing measurements.
///
/// The commitment arity is the trace's log-height plus the ceiling of the log of its width:
/// one extra variable per doubling of the column count, since every column is stacked into a
/// single committed polynomial.
///
/// The statement's security is assessed once against `options.security_bits` before proving,
/// so the timed phases are the plain prover and verifier.
///
/// The prover runs its zerocheck through [`SubfieldBackend`] over `GF(4)`, whose proof is
/// identical to the one [`p3_multi_stark::prove`] emits.
///
/// # Panics
///
/// - The trace height is not a power of two.
/// - `air` declares public values or preprocessed columns.
pub fn prove_binary_air<A>(
    air: &A,
    trace: RowMajorMatrix<F>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError>
where
    A: BinaryAir,
{
    assert_eq!(
        BaseAir::<F>::num_public_values(air),
        0,
        "the harness proves AIRs without public values"
    );
    assert_eq!(
        BaseAir::<F>::preprocessed_width(air),
        0,
        "the harness proves AIRs without preprocessed columns"
    );

    let rows = trace.height();
    let width = trace.width();
    let log_height = log2_strict_usize(rows);
    let arity = log_height + log2_ceil_usize(width);

    let params = BinaryPcsParams {
        log_inv_rate: options.log_inv_rate,
        pow_bits: options.pcs_pow_bits,
        security_level: options.security_bits,
    };
    let config = binary_config(arity, params, options.folding)?;

    let (pk, vk) = setup(&config, &[air], &mut binary_challenger())?;

    let public_values: [F; 0] = [];
    let verifier_instances = || {
        VerifierInstances::new(vec![VerifierInstance::new(
            air,
            &vk,
            log_height,
            &public_values,
        )])
    };

    let report =
        security_report(&config, &verifier_instances()).map_err(BinaryProofError::Security)?;
    report
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    let security_bits = report
        .security_bits()
        .expect("require_security succeeded, so every component is assessed");

    let prove_start = Instant::now();
    // Transpose into the sumcheck layout (one polynomial per row), then drop the row-major
    // trace so proving does not hold a second copy.
    let table = Table::new(trace.transpose());
    drop(trace);
    let prover_instances =
        ProverInstances::new(vec![ProverInstance::new(air, table, &pk, &public_values)]);
    let proof = prove_with_backend::<_, _, SubfieldBackend<BinaryField2>>(
        &config,
        prover_instances,
        options.sumcheck_pow_bits,
        &mut binary_challenger(),
    )?;
    let prove_seconds = prove_start.elapsed().as_secs_f64();

    let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
    let proof_bytes = bytes.len();
    let proof: MultiStarkProof<BinaryStarkConfig> =
        postcard::from_bytes(&bytes).expect("postcard round trip must not fail");

    let verify_start = Instant::now();
    verify(
        &config,
        verifier_instances(),
        &proof,
        options.sumcheck_pow_bits,
        &mut binary_challenger(),
    )?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();

    Ok(BinaryProofReport {
        rows,
        width,
        stacked_variables: arity,
        proof_bytes,
        prove_seconds,
        verify_seconds,
        security_bits,
    })
}

#[cfg(test)]
mod tests {
    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_binary_field::TowerLevel;
    use p3_blake3_air::Blake3BinaryAir;
    use p3_challenger::CanSample;
    use p3_field::HasSubfield;
    use p3_keccak_air::KeccakBinaryAir;
    use p3_multi_stark::prove;

    use super::*;

    /// A nonlinear recurrence over `BinaryField128`: `(a, b) -> (b, a * b + a)`.
    ///
    /// Addition is XOR and multiplication is tower-field multiplication. Nonlinear
    /// constraints exercise interpolation beyond the two prime-subfield elements.
    struct RecurrenceAir;

    impl<F> BaseAir<F> for RecurrenceAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB: AirBuilder> Air<AB> for RecurrenceAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            builder.when_transition().assert_eq(next[0], local[1]);
            builder
                .when_transition()
                .assert_eq(next[1], local[0] * local[1] + local[0]);
        }
    }

    fn recurrence_trace(log_height: usize) -> RowMajorMatrix<F> {
        let mut a = F::from_repr(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
        let mut b = F::from_repr(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef);
        let mut values = Vec::with_capacity(2 << log_height);
        for _ in 0..1 << log_height {
            values.extend([a, b]);
            (a, b) = (b, a * b + a);
        }
        RowMajorMatrix::new(values, 2)
    }

    #[test]
    fn proves_and_verifies_a_tiny_binary_air() {
        let log_height = 4;
        let trace = recurrence_trace(log_height);
        let report = prove_binary_air(&RecurrenceAir, trace, BinaryProofOptions::default())
            .expect("a tiny binary AIR proof must verify");
        assert_eq!(report.rows, 1 << log_height);
        assert_eq!(report.width, 2);
        assert_eq!(report.stacked_variables, log_height + 1);
        assert!(report.security_bits >= 100.0);
    }

    /// The serialized proof of `air` on `trace`, then the next challenge its transcript draws.
    ///
    /// With `subfield` the zerocheck runs through [`SubfieldBackend`], otherwise through [`prove`].
    fn proof_transcript<A: BinaryAir>(
        air: &A,
        trace: &RowMajorMatrix<F>,
        subfield: bool,
    ) -> (Vec<u8>, F) {
        let arity = log2_strict_usize(trace.height()) + log2_ceil_usize(trace.width());
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 100,
        };
        let config = binary_config(arity, params, 3).expect("the test shape configures the PCS");
        let (pk, _) = setup(&config, &[air], &mut binary_challenger()).expect("setup succeeds");

        let public_values: [F; 0] = [];
        let instances = ProverInstances::new(vec![ProverInstance::new(
            air,
            Table::new(trace.clone().transpose()),
            &pk,
            &public_values,
        )]);
        let mut challenger = binary_challenger();
        let proof = if subfield {
            prove_with_backend::<_, _, SubfieldBackend<BinaryField2>>(
                &config,
                instances,
                0,
                &mut challenger,
            )
        } else {
            prove(&config, instances, 0, &mut challenger)
        }
        .expect("an honest trace proves");
        let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
        (bytes, CanSample::<F>::sample(&mut challenger))
    }

    /// Whether every cell of `trace` lies in `GF(4)`, the cell condition for the subfield kernels.
    fn cells_fit_gf4(trace: &RowMajorMatrix<F>) -> bool {
        <F as HasSubfield<BinaryField2>>::all_in_subfield(&trace.values)
    }

    #[test]
    fn subfield_backend_proves_the_keccak_air_byte_for_byte() {
        // One permutation pads to 32 bit-valued rows of a degree-three AIR with successor columns.
        let air = KeccakBinaryAir {};
        let trace = air.generate_random_trace_rows::<F>(1, 0);
        assert_eq!(trace.height(), 32);
        assert!(cells_fit_gf4(&trace));
        assert_eq!(
            proof_transcript(&air, &trace, true),
            proof_transcript(&air, &trace, false)
        );
    }

    #[test]
    fn subfield_backend_proves_the_blake3_air_byte_for_byte() {
        // Four compressions, one bit-valued row each, of a degree-two AIR.
        let air = Blake3BinaryAir {};
        let trace = air.generate_random_trace_rows::<F>(4, 0);
        assert!(cells_fit_gf4(&trace));
        assert_eq!(
            proof_transcript(&air, &trace, true),
            proof_transcript(&air, &trace, false)
        );
    }

    #[test]
    fn subfield_backend_proves_a_full_width_trace_byte_for_byte() {
        // The recurrence starts from full-width cells, so its stage cannot fit `GF(4)` and the
        // subfield backend runs the generic kernels.
        let trace = recurrence_trace(4);
        assert!(!cells_fit_gf4(&trace));
        assert_eq!(
            proof_transcript(&RecurrenceAir, &trace, true),
            proof_transcript(&RecurrenceAir, &trace, false)
        );
    }
}
