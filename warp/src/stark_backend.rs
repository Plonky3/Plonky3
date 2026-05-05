//! Interop with OpenVM `stark-backend` committed SWIRL traces.
//!
//! This module is the handoff layer between OpenVM's segment prover and WARP.
//! It consumes [`CommittedProvingContext`] values produced by
//! `stark-backend`'s CPU backend after the common-main SWIRL commitment has
//! been built, but before the usual per-segment opening proof is generated.
//!
//! The important soundness boundary is that SWIRL's CPU PCS data is not a
//! [`p3_commit::Mmcs`] prover data object. This adapter therefore exposes the
//! committed Reed-Solomon codeword matrix and its native Merkle query material
//! directly instead of converting it into WARP's Plonky3-`Mmcs`
//! [`CommittedCodeword`](crate::protocol::CommittedCodeword).

use alloc::vec::Vec;

use openvm_cpu_backend::{CpuBackend, CpuStackedPcsData, merkle::CpuMerkleTree};
use openvm_stark_backend::{
    StarkProtocolConfig,
    hasher::MerkleHasher,
    proof::TraceVData,
    prover::{
        ColMajorMatrix, CommittedProvingContext, MatrixDimensions, error::StackedPcsError,
        stacked_pcs::StackedLayout,
    },
    verifier::whir::{VerifyWhirError, merkle_verify},
};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::protocol::{
    AccumulatorCommitmentBackend, ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier,
    ExternalCommitmentObserver, ExternalCommittedCodeword,
};

/// A committed OpenVM/SWIRL proving context using the CPU backend.
pub type CpuCommittedProvingContext<SC> = CommittedProvingContext<SC, CpuBackend<SC>>;

/// CPU PCS data used by `stark-backend` for a committed segment.
pub type StarkBackendCpuPcsData<SC> =
    CpuStackedPcsData<<SC as StarkProtocolConfig>::F, <SC as StarkProtocolConfig>::Digest>;

/// Trace shape and cached-preprocessing commitments, in WARP-owned form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Digest: Serialize + serde::de::DeserializeOwned")]
pub struct StarkBackendTraceVData<Digest> {
    /// Base-2 logarithm of the trace height.
    pub log_height: usize,
    /// Cached commitments referenced by the trace.
    pub cached_commitments: Vec<Digest>,
}

/// One stacked column entry from `stark-backend`'s SWIRL layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkBackendStackedColumn {
    /// Index of the original unstacked matrix.
    pub matrix_index: usize,
    /// Column index inside the original unstacked matrix.
    pub column_index: usize,
    /// Column index inside the stacked matrix.
    pub stacked_column_index: usize,
    /// First row occupied by this slice inside the stacked matrix.
    pub stacked_row_index: usize,
    /// Base-2 logarithm of the original column height.
    pub log_height: usize,
}

/// Public shape of the SWIRL stacked matrix layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkBackendStackedLayout {
    /// Minimum log-height used when stacking short traces.
    pub l_skip: usize,
    /// Height of the stacked evaluation matrix.
    pub height: usize,
    /// Width of the stacked evaluation matrix.
    pub width: usize,
    /// Start offsets in `sorted_columns`, indexed by original matrix.
    pub matrix_starts: Vec<usize>,
    /// Sorted column placement metadata.
    pub sorted_columns: Vec<StarkBackendStackedColumn>,
}

impl From<&StackedLayout> for StarkBackendStackedLayout {
    fn from(layout: &StackedLayout) -> Self {
        let sorted_columns = layout
            .sorted_cols
            .iter()
            .map(
                |(matrix_index, column_index, slice)| StarkBackendStackedColumn {
                    matrix_index: *matrix_index,
                    column_index: *column_index,
                    stacked_column_index: slice.col_idx,
                    stacked_row_index: slice.row_idx,
                    log_height: slice.log_height(),
                },
            )
            .collect();

        Self {
            l_skip: layout.l_skip(),
            height: layout.height(),
            width: layout.width(),
            matrix_starts: layout.mat_starts.clone(),
            sorted_columns,
        }
    }
}

/// Verifier-visible metadata for a committed SWIRL segment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  Digest: Serialize + serde::de::DeserializeOwned")]
pub struct StarkBackendSegmentClaim<F, Digest> {
    /// SWIRL common-main Merkle commitment.
    pub common_main_commit: Digest,
    /// Trace metadata in AIR verifying-key order.
    pub trace_vdata: Vec<Option<StarkBackendTraceVData<Digest>>>,
    /// Public values in AIR verifying-key order.
    pub public_values: Vec<Vec<F>>,
    /// Stacked-evaluation layout committed by SWIRL.
    pub layout: StarkBackendStackedLayout,
    /// Height of the stacked evaluation matrix before RS encoding.
    pub stacked_eval_height: usize,
    /// Width of the stacked evaluation matrix before RS encoding.
    pub stacked_eval_width: usize,
    /// Height of the committed Reed-Solomon codeword matrix.
    pub rs_codeword_height: usize,
    /// Width of the committed Reed-Solomon codeword matrix.
    pub rs_codeword_width: usize,
    /// Number of RS-codeword rows opened per query.
    pub rows_per_query: usize,
    /// Number of Merkle leaves at the query layer.
    pub query_stride: usize,
    /// Number of sibling hashes in each Merkle authentication path.
    pub merkle_proof_depth: usize,
}

/// SWIRL initial-round opening proof for one flattened WARP codeword entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  Digest: Serialize + serde::de::DeserializeOwned")]
pub struct StarkBackendOpeningProof<F, Digest> {
    /// SWIRL query index, i.e. row modulo `query_stride`.
    pub query_index: usize,
    /// Row offset inside the authenticated row group.
    pub row_offset: usize,
    /// Column containing the flattened WARP answer.
    pub column_index: usize,
    /// Authenticated row group returned by SWIRL.
    pub opened_rows: Vec<Vec<F>>,
    /// Merkle authentication path for the row-group digest.
    pub merkle_proof: Vec<Digest>,
}

/// Prover-side SWIRL opening backend for [`StarkBackendSegment`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StarkBackendOpeningBackend;

/// Verifier-side SWIRL opening backend.
#[derive(Clone, Copy, Debug)]
pub struct StarkBackendOpeningVerifier<'a, SC: StarkProtocolConfig> {
    config: &'a SC,
}

impl<'a, SC: StarkProtocolConfig> StarkBackendOpeningVerifier<'a, SC> {
    /// Create a verifier using the same SWIRL configuration that produced the
    /// committed segment.
    pub fn new(config: &'a SC) -> Self {
        Self { config }
    }
}

/// Verifier-visible SWIRL-style commitment for a WARP accumulator codeword.
///
/// WARP accumulators live over the challenge extension field `EF`. SWIRL's
/// initial commitments are Merkle roots over base-field rows. This claim uses
/// the same hash convention by hashing each `EF` row through its canonical
/// base-field coefficients, while keeping the committed codeword logically over
/// `EF`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "Digest: Serialize + serde::de::DeserializeOwned")]
pub struct StarkBackendAccumulatorClaim<Digest> {
    /// Merkle root of the accumulator codeword.
    pub commitment: Digest,
    /// Base-2 logarithm of the committed WARP codeword length.
    pub log_codeword_len: usize,
    /// Degree of `EF` as a vector space over the SWIRL base field.
    pub extension_degree: usize,
    /// Number of codeword rows authenticated by one SWIRL query.
    pub rows_per_query: usize,
    /// Number of query positions in the bottom Merkle layer.
    pub query_stride: usize,
    /// Number of sibling hashes in each Merkle authentication path.
    pub merkle_proof_depth: usize,
}

/// Prover-side data for a SWIRL-style WARP accumulator commitment.
#[derive(Clone, Debug)]
pub struct StarkBackendAccumulatorProverData<EF, Digest> {
    tree: CpuMerkleTree<EF, Digest>,
}

impl<EF, Digest> StarkBackendAccumulatorProverData<EF, Digest> {
    /// Return the committed row-major accumulator codeword.
    pub fn codeword_matrix(&self) -> &RowMajorMatrix<EF> {
        self.tree.backing_matrix()
    }
}

/// SWIRL-style Merkle opening for one WARP accumulator codeword entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned,
                  Digest: Serialize + serde::de::DeserializeOwned")]
pub struct StarkBackendAccumulatorOpeningProof<EF, Digest> {
    /// Query index in the bottom Merkle layer.
    pub query_index: usize,
    /// Row offset inside the authenticated row group.
    pub row_offset: usize,
    /// Authenticated extension-field rows for this query.
    pub opened_values: Vec<EF>,
    /// Merkle authentication path for the row-group digest.
    pub merkle_proof: Vec<Digest>,
}

/// SWIRL-compatible commitment/opening backend for WARP accumulator codewords.
#[derive(Clone, Copy, Debug)]
pub struct StarkBackendAccumulatorBackend<'a, SC: StarkProtocolConfig> {
    config: &'a SC,
    rows_per_query: usize,
}

impl<'a, SC> StarkBackendAccumulatorBackend<'a, SC>
where
    SC: StarkProtocolConfig,
{
    /// Use the same row batching as the configured WHIR initial Merkle layer.
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            rows_per_query: 1usize << config.params().k_whir(),
        }
    }

    /// Override the number of rows per Merkle query. This is useful for small
    /// tests and for experiments that tune WARP independently from full SWIRL.
    pub fn with_rows_per_query(config: &'a SC, rows_per_query: usize) -> Self {
        Self {
            config,
            rows_per_query,
        }
    }

    /// Commit an extension-field WARP accumulator codeword with SWIRL's base
    /// hasher, hashing each extension value as its base-field coefficient row.
    pub fn commit(
        &self,
        codeword: Vec<SC::EF>,
    ) -> Result<
        (
            StarkBackendAccumulatorClaim<SC::Digest>,
            StarkBackendAccumulatorProverData<SC::EF, SC::Digest>,
        ),
        StarkBackendExternalOpeningError,
    > {
        let len = codeword.len();
        if len == 0 || !len.is_power_of_two() {
            return Err(StarkBackendExternalOpeningError::CodewordLength {
                got: len,
                expected: len.next_power_of_two(),
            });
        }
        let tree =
            build_accumulator_tree::<SC>(self.config.hasher(), codeword, self.rows_per_query)?;
        let commitment = tree.root()?;
        let claim = StarkBackendAccumulatorClaim {
            commitment,
            log_codeword_len: len.trailing_zeros() as usize,
            extension_degree: SC::D_EF,
            rows_per_query: tree.rows_per_query(),
            query_stride: tree.query_stride(),
            merkle_proof_depth: tree.proof_depth(),
        };
        Ok((claim, StarkBackendAccumulatorProverData { tree }))
    }

    /// Open one accumulator codeword entry.
    pub fn open(
        &self,
        prover_data: &StarkBackendAccumulatorProverData<SC::EF, SC::Digest>,
        index: usize,
    ) -> Result<
        (
            SC::EF,
            StarkBackendAccumulatorOpeningProof<SC::EF, SC::Digest>,
        ),
        StarkBackendExternalOpeningError,
    > {
        let height = prover_data.tree.backing_matrix().height();
        let claim = StarkBackendAccumulatorClaim {
            commitment: prover_data.tree.root()?,
            log_codeword_len: height.trailing_zeros() as usize,
            extension_degree: SC::D_EF,
            rows_per_query: prover_data.tree.rows_per_query(),
            query_stride: prover_data.tree.query_stride(),
            merkle_proof_depth: prover_data.tree.proof_depth(),
        };
        let (query_index, row_offset) = accumulator_opening_position(&claim, index, None)?;
        let opened_rows = prover_data.tree.get_opened_rows(query_index)?;
        let opened_values: Vec<SC::EF> = opened_rows
            .into_iter()
            .map(|mut row| {
                debug_assert_eq!(row.len(), 1);
                row.swap_remove(0)
            })
            .collect();
        let value =
            *opened_values
                .get(row_offset)
                .ok_or(StarkBackendExternalOpeningError::RowGroup(
                    "row offset out of bounds",
                ))?;
        let merkle_proof = prover_data.tree.query_merkle_proof(query_index)?;
        Ok((
            value,
            StarkBackendAccumulatorOpeningProof {
                query_index,
                row_offset,
                opened_values,
                merkle_proof,
            },
        ))
    }

    /// Absorb the accumulator commitment metadata into a WARP transcript.
    pub fn observe_commitment<Challenger>(
        &self,
        challenger: &mut Challenger,
        claim: &StarkBackendAccumulatorClaim<SC::Digest>,
    ) where
        Challenger: FieldChallenger<SC::F> + CanObserve<SC::Digest>,
    {
        challenger.observe(claim.commitment);
        observe_usize(challenger, claim.log_codeword_len);
        observe_usize(challenger, claim.extension_degree);
        observe_usize(challenger, claim.rows_per_query);
        observe_usize(challenger, claim.query_stride);
        observe_usize(challenger, claim.merkle_proof_depth);
    }

    /// Verify one SWIRL-style accumulator opening.
    pub fn verify_opening(
        &self,
        claim: &StarkBackendAccumulatorClaim<SC::Digest>,
        expected_log_codeword_len: usize,
        index: usize,
        value: SC::EF,
        proof: &StarkBackendAccumulatorOpeningProof<SC::EF, SC::Digest>,
    ) -> Result<(), StarkBackendExternalOpeningError> {
        let (query_index, row_offset) =
            accumulator_opening_position(claim, index, Some(expected_log_codeword_len))?;
        if claim.extension_degree != SC::D_EF {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "extension degree mismatch",
            ));
        }
        if proof.query_index != query_index {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "query index mismatch",
            ));
        }
        if proof.row_offset != row_offset {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "row offset mismatch",
            ));
        }
        if proof.opened_values.len() != claim.rows_per_query {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "opened row count mismatch",
            ));
        }
        if proof.merkle_proof.len() != claim.merkle_proof_depth {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "merkle proof depth mismatch",
            ));
        }
        if proof.opened_values[row_offset] != value {
            return Err(StarkBackendExternalOpeningError::ValueMismatch);
        }

        let hasher = self.config.hasher();
        let leaf_hashes = proof
            .opened_values
            .iter()
            .map(|opened| hasher.hash_slice(opened.as_basis_coefficients_slice()))
            .collect();
        let query_digest = hasher.tree_compress(leaf_hashes);
        merkle_verify(
            hasher,
            claim.commitment,
            query_index as u32,
            query_digest,
            &proof.merkle_proof,
        )?;
        Ok(())
    }
}

impl<'a, SC, Challenger> AccumulatorCommitmentBackend<SC::F, SC::EF, Challenger>
    for StarkBackendAccumulatorBackend<'a, SC>
where
    SC: StarkProtocolConfig,
    Challenger: FieldChallenger<SC::F> + CanObserve<SC::Digest>,
{
    type Commitment = StarkBackendAccumulatorClaim<SC::Digest>;
    type ProverData = StarkBackendAccumulatorProverData<SC::EF, SC::Digest>;
    type Proof = StarkBackendAccumulatorOpeningProof<SC::EF, SC::Digest>;
    type Error = StarkBackendExternalOpeningError;

    fn commit(
        &self,
        codeword: Vec<SC::EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        StarkBackendAccumulatorBackend::commit(self, codeword)
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(SC::EF, Self::Proof), Self::Error> {
        StarkBackendAccumulatorBackend::open(self, prover_data, index)
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        StarkBackendAccumulatorBackend::observe_commitment(self, challenger, commitment);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: SC::EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        StarkBackendAccumulatorBackend::verify_opening(
            self,
            commitment,
            log_codeword_len,
            index,
            value,
            proof,
        )
    }
}

/// Errors while opening or verifying a `stark-backend` external codeword.
#[derive(Error, Debug)]
pub enum StarkBackendExternalOpeningError {
    /// The flattened committed matrix does not match WARP's configured `n`.
    #[error("external codeword length mismatch: got {got}, expected {expected}")]
    CodewordLength { got: usize, expected: usize },

    /// The committed RS codeword matrix has zero width.
    #[error("external codeword matrix has zero width")]
    EmptyWidth,

    /// The SWIRL/WARP query row grouping is invalid.
    #[error("invalid rows_per_query {rows_per_query} for {num_rows} committed rows")]
    RowsPerQuery {
        rows_per_query: usize,
        num_rows: usize,
    },

    /// The queried row group is inconsistent with the claim.
    #[error("external opening row group mismatch: {0}")]
    RowGroup(&'static str),

    /// The scalar WARP answer is not the authenticated cell.
    #[error("external opening value mismatch")]
    ValueMismatch,

    /// Underlying SWIRL PCS opening error.
    #[error("stacked PCS error: {0}")]
    StackedPcs(#[from] StackedPcsError),

    /// Underlying SWIRL Merkle verification error.
    #[error("SWIRL Merkle verification error: {0}")]
    Whir(#[from] VerifyWhirError),
}

/// Borrowed view over a `stark-backend` CPU committed proving context.
#[derive(Clone, Copy)]
pub struct StarkBackendSegment<'a, SC: StarkProtocolConfig> {
    committed: &'a CpuCommittedProvingContext<SC>,
}

impl<'a, SC> StarkBackendSegment<'a, SC>
where
    SC: StarkProtocolConfig,
{
    /// Create a WARP handoff view over a committed SWIRL segment.
    pub fn new(committed: &'a CpuCommittedProvingContext<SC>) -> Self {
        Self { committed }
    }

    /// Return the original committed proving context.
    pub fn committed_context(&self) -> &'a CpuCommittedProvingContext<SC> {
        self.committed
    }

    /// Return the CPU PCS data produced by `stark-backend`.
    pub fn pcs_data(&self) -> &'a StarkBackendCpuPcsData<SC> {
        &self.committed.common_main_pcs_data
    }

    /// Return the stacked evaluation matrix before RS encoding.
    pub fn stacked_eval_matrix(&self) -> &'a ColMajorMatrix<SC::F> {
        &self.pcs_data().matrix
    }

    /// Return the row-major Reed-Solomon codeword matrix committed by SWIRL.
    pub fn rs_codeword_matrix(&self) -> &'a RowMajorMatrix<SC::F> {
        self.pcs_data().rs_codeword_matrix()
    }

    /// Return the flattened RS codeword values in row-major order.
    pub fn rs_codeword_values(&self) -> &'a [SC::F] {
        &self.rs_codeword_matrix().values
    }

    /// Return the Merkle authentication path for a SWIRL query index.
    pub fn merkle_proof(&self, query_idx: usize) -> Result<Vec<SC::Digest>, StackedPcsError> {
        self.pcs_data().tree.query_merkle_proof(query_idx)
    }

    /// Return the RS-codeword rows opened by a SWIRL query index.
    pub fn opened_rows(&self, query_idx: usize) -> Result<Vec<Vec<SC::F>>, StackedPcsError> {
        self.pcs_data().tree.get_opened_rows(query_idx)
    }

    /// Recompute the root from the PCS data and compare it to the exported
    /// common-main commitment.
    pub fn commitment_is_consistent(&self) -> Result<bool, StackedPcsError> {
        Ok(self.pcs_data().commit()? == self.committed.common_main_commit)
    }

    /// Build the WARP-side claim metadata for this committed segment.
    pub fn claim(&self) -> StarkBackendSegmentClaim<SC::F, SC::Digest> {
        let pcs = self.pcs_data();
        let rs_codeword = pcs.rs_codeword_matrix();
        let trace_vdata = self
            .committed
            .trace_vdata
            .iter()
            .map(trace_vdata_to_warp)
            .collect();

        StarkBackendSegmentClaim {
            common_main_commit: self.committed.common_main_commit,
            trace_vdata,
            public_values: self.committed.public_values.clone(),
            layout: StarkBackendStackedLayout::from(&pcs.layout),
            stacked_eval_height: pcs.matrix.height(),
            stacked_eval_width: pcs.matrix.width(),
            rs_codeword_height: rs_codeword.values.len() / rs_codeword.width,
            rs_codeword_width: rs_codeword.width,
            rows_per_query: pcs.tree.rows_per_query(),
            query_stride: pcs.tree.query_stride(),
            merkle_proof_depth: pcs.tree.proof_depth(),
        }
    }
}

fn trace_vdata_to_warp<SC>(
    trace_vdata: &Option<TraceVData<SC>>,
) -> Option<StarkBackendTraceVData<SC::Digest>>
where
    SC: StarkProtocolConfig,
{
    trace_vdata.as_ref().map(|trace| StarkBackendTraceVData {
        log_height: trace.log_height,
        cached_commitments: trace.cached_commitments.clone(),
    })
}

impl<'a, SC> ExternalCommittedCodeword<SC::F> for StarkBackendSegment<'a, SC>
where
    SC: StarkProtocolConfig,
{
    type Commitment = StarkBackendSegmentClaim<SC::F, SC::Digest>;

    fn commitment(&self) -> Self::Commitment {
        self.claim()
    }

    fn codeword(&self) -> &[SC::F] {
        self.rs_codeword_values()
    }

    fn witness(&self) -> &[SC::F] {
        &self.stacked_eval_matrix().values
    }
}

impl<'a, SC, Challenger> ExternalCommitmentObserver<SC::F, Challenger>
    for StarkBackendSegment<'a, SC>
where
    SC: StarkProtocolConfig,
    Challenger: FieldChallenger<SC::F> + CanObserve<SC::Digest>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        observe_claim::<SC, Challenger>(challenger, &self.claim());
    }
}

impl<'a, SC> ExternalCodewordOpeningProver<SC::F, StarkBackendSegment<'a, SC>>
    for StarkBackendOpeningBackend
where
    SC: StarkProtocolConfig,
{
    type Proof = StarkBackendOpeningProof<SC::F, SC::Digest>;
    type Error = StarkBackendExternalOpeningError;

    fn open(
        &self,
        committed: &StarkBackendSegment<'a, SC>,
        index: usize,
    ) -> Result<(SC::F, Self::Proof), Self::Error> {
        let claim = committed.claim();
        let (query_index, row_offset, column_index) = opening_position(&claim, index, None)?;
        let opened_rows = committed.opened_rows(query_index)?;
        let row = opened_rows
            .get(row_offset)
            .ok_or(StarkBackendExternalOpeningError::RowGroup(
                "row offset out of bounds",
            ))?;
        let value = *row
            .get(column_index)
            .ok_or(StarkBackendExternalOpeningError::RowGroup(
                "column index out of bounds",
            ))?;
        let merkle_proof = committed.merkle_proof(query_index)?;

        Ok((
            value,
            StarkBackendOpeningProof {
                query_index,
                row_offset,
                column_index,
                opened_rows,
                merkle_proof,
            },
        ))
    }
}

impl<'a, SC, Challenger> ExternalCodewordOpeningVerifier<SC::F, Challenger>
    for StarkBackendOpeningVerifier<'a, SC>
where
    SC: StarkProtocolConfig,
    Challenger: FieldChallenger<SC::F> + CanObserve<SC::Digest>,
{
    type Commitment = StarkBackendSegmentClaim<SC::F, SC::Digest>;
    type Proof = StarkBackendOpeningProof<SC::F, SC::Digest>;
    type Error = StarkBackendExternalOpeningError;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        observe_claim::<SC, Challenger>(challenger, commitment);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: SC::F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let (query_index, row_offset, column_index) =
            opening_position(commitment, index, Some(log_codeword_len))?;
        if proof.query_index != query_index {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "query index mismatch",
            ));
        }
        if proof.row_offset != row_offset {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "row offset mismatch",
            ));
        }
        if proof.column_index != column_index {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "column index mismatch",
            ));
        }
        if proof.opened_rows.len() != commitment.rows_per_query {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "opened row count mismatch",
            ));
        }
        if proof.merkle_proof.len() != commitment.merkle_proof_depth {
            return Err(StarkBackendExternalOpeningError::RowGroup(
                "merkle proof depth mismatch",
            ));
        }
        let row =
            proof
                .opened_rows
                .get(row_offset)
                .ok_or(StarkBackendExternalOpeningError::RowGroup(
                    "row offset out of bounds",
                ))?;
        let authenticated =
            row.get(column_index)
                .ok_or(StarkBackendExternalOpeningError::RowGroup(
                    "column index out of bounds",
                ))?;
        if *authenticated != value {
            return Err(StarkBackendExternalOpeningError::ValueMismatch);
        }

        let hasher = self.config.hasher();
        let leaf_hashes = proof
            .opened_rows
            .iter()
            .map(|opened_row| hasher.hash_slice(opened_row))
            .collect();
        let query_digest = hasher.tree_compress(leaf_hashes);
        merkle_verify(
            hasher,
            commitment.common_main_commit,
            query_index as u32,
            query_digest,
            &proof.merkle_proof,
        )?;
        Ok(())
    }
}

fn opening_position<F, Digest>(
    claim: &StarkBackendSegmentClaim<F, Digest>,
    index: usize,
    expected_log_len: Option<usize>,
) -> Result<(usize, usize, usize), StarkBackendExternalOpeningError> {
    if claim.rs_codeword_width == 0 {
        return Err(StarkBackendExternalOpeningError::EmptyWidth);
    }
    let got = claim.rs_codeword_height * claim.rs_codeword_width;
    if let Some(log_len) = expected_log_len {
        let expected = 1usize << log_len;
        if got != expected {
            return Err(StarkBackendExternalOpeningError::CodewordLength { got, expected });
        }
    }
    if index >= got {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "flattened index out of bounds",
        ));
    }
    let row = index / claim.rs_codeword_width;
    let column = index % claim.rs_codeword_width;
    if claim.query_stride == 0 {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "query stride is zero",
        ));
    }
    let query_index = row % claim.query_stride;
    let row_offset = row / claim.query_stride;
    if row_offset >= claim.rows_per_query {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "row offset exceeds rows_per_query",
        ));
    }
    Ok((query_index, row_offset, column))
}

fn observe_claim<SC, Challenger>(
    challenger: &mut Challenger,
    claim: &StarkBackendSegmentClaim<SC::F, SC::Digest>,
) where
    SC: StarkProtocolConfig,
    Challenger: FieldChallenger<SC::F> + CanObserve<SC::Digest>,
{
    challenger.observe(claim.common_main_commit);
    observe_usize(challenger, claim.layout.l_skip);
    observe_usize(challenger, claim.layout.height);
    observe_usize(challenger, claim.layout.width);
    observe_usize(challenger, claim.stacked_eval_height);
    observe_usize(challenger, claim.stacked_eval_width);
    observe_usize(challenger, claim.rs_codeword_height);
    observe_usize(challenger, claim.rs_codeword_width);
    observe_usize(challenger, claim.rows_per_query);
    observe_usize(challenger, claim.query_stride);
    observe_usize(challenger, claim.merkle_proof_depth);

    observe_usize(challenger, claim.trace_vdata.len());
    for trace in &claim.trace_vdata {
        challenger.observe(SC::F::from_bool(trace.is_some()));
        if let Some(trace) = trace {
            observe_usize(challenger, trace.log_height);
            observe_usize(challenger, trace.cached_commitments.len());
            for &commitment in &trace.cached_commitments {
                challenger.observe(commitment);
            }
        }
    }

    observe_usize(challenger, claim.public_values.len());
    for values in &claim.public_values {
        observe_usize(challenger, values.len());
        challenger.observe_slice(values);
    }
}

fn observe_usize<F, Challenger>(challenger: &mut Challenger, value: usize)
where
    F: Field + PrimeCharacteristicRing,
    Challenger: FieldChallenger<F>,
{
    challenger.observe(F::from_usize(value));
}

fn accumulator_opening_position<Digest>(
    claim: &StarkBackendAccumulatorClaim<Digest>,
    index: usize,
    expected_log_len: Option<usize>,
) -> Result<(usize, usize), StarkBackendExternalOpeningError> {
    if let Some(expected_log_len) = expected_log_len {
        let expected = 1usize << expected_log_len;
        let got = 1usize << claim.log_codeword_len;
        if got != expected {
            return Err(StarkBackendExternalOpeningError::CodewordLength { got, expected });
        }
    }
    let len = 1usize << claim.log_codeword_len;
    if index >= len {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "accumulator opening index out of bounds",
        ));
    }
    if claim.query_stride == 0 || claim.rows_per_query == 0 {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "empty accumulator query layout",
        ));
    }
    if claim.query_stride * claim.rows_per_query != len {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "accumulator query layout does not match codeword length",
        ));
    }
    let query_index = index % claim.query_stride;
    let row_offset = index / claim.query_stride;
    if row_offset >= claim.rows_per_query {
        return Err(StarkBackendExternalOpeningError::RowGroup(
            "row offset exceeds rows_per_query",
        ));
    }
    Ok((query_index, row_offset))
}

fn build_accumulator_tree<SC>(
    hasher: &SC::Hasher,
    codeword: Vec<SC::EF>,
    rows_per_query: usize,
) -> Result<CpuMerkleTree<SC::EF, SC::Digest>, StarkBackendExternalOpeningError>
where
    SC: StarkProtocolConfig,
{
    let num_rows = codeword.len();
    if rows_per_query == 0
        || !rows_per_query.is_power_of_two()
        || rows_per_query > num_rows
        || !num_rows.is_power_of_two()
    {
        return Err(StarkBackendExternalOpeningError::RowsPerQuery {
            rows_per_query,
            num_rows,
        });
    }

    let query_stride = num_rows / rows_per_query;
    let row_hashes: Vec<SC::Digest> = codeword
        .iter()
        .map(|value| hasher.hash_slice(value.as_basis_coefficients_slice()))
        .collect();

    let mut bottom = Vec::with_capacity(query_stride);
    for query_index in 0..query_stride {
        let leaves = (0..rows_per_query)
            .map(|row_offset| row_hashes[row_offset * query_stride + query_index])
            .collect();
        bottom.push(hasher.tree_compress(leaves));
    }

    let mut layers = Vec::new();
    layers.push(bottom);
    while layers.last().expect("bottom layer exists").len() > 1 {
        let prev = layers.last().expect("previous layer exists");
        let next = prev
            .chunks_exact(2)
            .map(|pair| hasher.compress(pair[0], pair[1]))
            .collect();
        layers.push(next);
    }

    let backing_matrix = RowMajorMatrix::new(codeword, 1);
    // SAFETY: `layers` are computed immediately above from `backing_matrix`
    // using the same strided row grouping as stark-backend's CPU Merkle tree.
    Ok(unsafe { CpuMerkleTree::from_raw_parts(backing_matrix, layers, rows_per_query) })
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::{
        app_params_with_100_bits_security,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F},
    };
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn test_config() -> BabyBearPoseidon2Config {
        BabyBearPoseidon2Config::default_from_params(app_params_with_100_bits_security(5))
    }

    #[test]
    fn accumulator_backend_opens_and_verifies() {
        let config = test_config();
        let backend = StarkBackendAccumulatorBackend::with_rows_per_query(&config, 4);
        let codeword: Vec<EF> = (0..32).map(|i| EF::from_u64((17 * i + 3) as u64)).collect();

        let (claim, prover_data) = backend.commit(codeword.clone()).unwrap();
        assert_eq!(claim.log_codeword_len, 5);
        assert_eq!(claim.extension_degree, 4);
        assert_eq!(claim.rows_per_query, 4);
        assert_eq!(claim.query_stride, 8);

        let (value, proof) = backend.open(&prover_data, 19).unwrap();
        assert_eq!(value, codeword[19]);
        backend
            .verify_opening(&claim, 5, 19, value, &proof)
            .expect("opening verifies");
    }

    #[test]
    fn accumulator_backend_rejects_tampered_value() {
        let config = test_config();
        let backend = StarkBackendAccumulatorBackend::with_rows_per_query(&config, 4);
        let codeword: Vec<EF> = (0..32).map(|i| EF::from_u64((5 * i + 1) as u64)).collect();

        let (claim, prover_data) = backend.commit(codeword).unwrap();
        let (value, proof) = backend.open(&prover_data, 7).unwrap();
        let err = backend
            .verify_opening(&claim, 5, 7, value + EF::ONE, &proof)
            .expect_err("tampered value must fail");
        assert!(matches!(
            err,
            StarkBackendExternalOpeningError::ValueMismatch
        ));
    }

    #[test]
    fn accumulator_backend_rejects_tampered_opened_row() {
        let config = test_config();
        let backend = StarkBackendAccumulatorBackend::with_rows_per_query(&config, 4);
        let codeword: Vec<EF> = (0..32).map(|i| EF::from_u64((11 * i + 9) as u64)).collect();

        let (claim, prover_data) = backend.commit(codeword).unwrap();
        let (value, mut proof) = backend.open(&prover_data, 23).unwrap();
        proof.opened_values[proof.row_offset] += EF::from(F::ONE);
        let err = backend
            .verify_opening(&claim, 5, 23, value, &proof)
            .expect_err("tampered authenticated row must fail");
        assert!(matches!(
            err,
            StarkBackendExternalOpeningError::ValueMismatch
                | StarkBackendExternalOpeningError::Whir(_)
        ));
    }
}
