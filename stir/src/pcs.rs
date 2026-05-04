//! `TwoAdicStirPcs`: implementing the [`Pcs`] trait using STIR.
//!
//! **Commit**: given trace matrices evaluated on two-adic cosets, compute their LDEs and commit
//! to those LDEs using an `InputMmcs`.
//!
//! **Open**: alpha-batch quotient polynomials `(f_i(z) - f_i(x)) / (z - x)` into per-height
//! reduced-opening polynomials, then run [`prove_stir`] on each distinct LDE-height bucket.
//! At each bucket's first-round STIR query positions the prover also opens the input LDE
//! matrices (via `InputMmcs`) so the verifier can confirm the reduced-opening polynomial is
//! correctly derived from the committed inputs.
//!
//! **Verify**: replay the same alpha-batching from the opening values, then for each height
//! bucket call [`verify_stir`] and replay the first-round transcript to derive query positions,
//! verifying the input MMCS openings against the committed input and checking consistency with
//! the STIR first-round fiber evaluations.

use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{
    BatchOpening, BatchOpeningRef, BuildPeriodicLdeTableFast, Mmcs, OpenedValues, Pcs,
};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    BasedVectorSpace, ExtensionField, PackedFieldExtension, TwoAdicField,
    batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixCow, RowMajorMatrixView};
use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
use tracing::instrument;

use crate::config::{StirConfig, StirParameters};
use crate::proof::StirProof;
use crate::prover::{coeffs_from_codeword, prove_stir};
use crate::verifier::{StirError, verify_stir};

/// A polynomial commitment scheme using STIR to generate opening proofs.
#[derive(Clone, Debug)]
pub struct TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs> {
    dft: Dft,
    input_mmcs: InputMmcs,
    stir: StirParameters<StirMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, StirMmcs> TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs> {
    pub const fn new(dft: Dft, input_mmcs: InputMmcs, stir: StirParameters<StirMmcs>) -> Self {
        Self {
            dft,
            input_mmcs,
            stir,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Dft, InputMmcs, StirMmcs> BuildPeriodicLdeTableFast
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs>
where
    Val: TwoAdicField,
{
    type PeriodicDomain = TwoAdicMultiplicativeCoset<Val>;
}

impl<Val, Dft, InputMmcs, StirMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for TwoAdicStirPcs<Val, Dft, InputMmcs, StirMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val, Proof: Sync, Error: Sync + Debug>,
    StirMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val> + TwoAdicField + BasedVectorSpace<Val>,
    Challenger: FieldChallenger<Val>
        + CanObserve<InputMmcs::Commitment>
        + CanObserve<StirMmcs::Commitment>
        + GrindingChallenger<Witness = Val>
        + Clone,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixCow<'a, Val>>;
    /// Proof structure: one entry per distinct LDE-height bucket (descending).
    ///
    /// Each bucket contains:
    /// - `stir_proof`: the STIR IOP proof for that bucket (includes the initial commitment
    ///   and first-round query indices).
    /// - `input_openings[commit_idx]`: per-commitment batch openings at the
    ///   bucket's first-round STIR fiber positions.  Empty if the commitment
    ///   has no matrices at this bucket's LDE height.
    type Proof = Vec<(
        StirProof<Challenge, StirMmcs, Val>,
        Vec<Vec<BatchOpening<Val, InputMmcs>>>,
    )>;
    type Error = StirError<StirMmcs::Error, InputMmcs::Error>;

    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let min_height = 1usize << self.stir.log_folding_factor;
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                assert!(
                    evals.height() >= min_height,
                    "STIR PCS: matrix height {} is below the minimum of 2^{} (= {}) required \
                     by log_folding_factor = {}. Pad the matrix to at least this height before \
                     committing, or lower log_folding_factor.",
                    evals.height(),
                    self.stir.log_folding_factor,
                    min_height,
                    self.stir.log_folding_factor,
                );
                let shift = Val::GENERATOR / domain.shift();
                self.dft
                    .coset_lde_batch(evals, self.stir.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();
        self.input_mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        let lde = self.input_mmcs.get_matrices(prover_data)[idx];
        if domain.shift() == Val::GENERATOR && lde.height() >= domain.size() {
            return lde.split_rows(domain.size()).0.as_cow().bit_reverse_rows();
        }
        let poly_height = lde.height() >> self.stir.log_blowup;
        let lde_mat = lde.as_view().bit_reverse_rows().to_row_major_matrix();
        let mut coeffs = self.dft.coset_idft_batch(lde_mat, Val::GENERATOR);
        let width = coeffs.width();
        coeffs.values.truncate(poly_height * width);
        coeffs.values.resize(domain.size() * width, Val::ZERO);
        let result = self
            .dft
            .coset_dft_batch(coeffs, domain.shift())
            .to_row_major_matrix();
        let result_width = result.width();
        RowMajorMatrixCow::new(Cow::Owned(result.values), result_width).bit_reverse_rows()
    }

    fn get_quotient_ldes(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
        _num_chunks: usize,
    ) -> Vec<RowMajorMatrix<Val>> {
        let min_height = 1usize << self.stir.log_folding_factor;
        evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert!(
                    evals.height() >= min_height,
                    "STIR PCS quotient: matrix height {} is below 2^{} required by \
                     log_folding_factor = {}.",
                    evals.height(),
                    self.stir.log_folding_factor,
                    self.stir.log_folding_factor,
                );
                let shift = Val::GENERATOR / domain.shift();
                self.dft
                    .coset_lde_batch(evals, self.stir.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn commit_ldes(&self, ldes: Vec<RowMajorMatrix<Val>>) -> (Self::Commitment, Self::ProverData) {
        let min_lde_height = 1usize << (self.stir.log_folding_factor + self.stir.log_blowup);
        for lde in &ldes {
            assert!(
                lde.height() >= min_lde_height,
                "STIR PCS: pre-computed LDE height {} is below 2^{} (= {}) required by \
                 log_folding_factor + log_blowup = {} + {}.",
                lde.height(),
                self.stir.log_folding_factor + self.stir.log_blowup,
                min_lde_height,
                self.stir.log_folding_factor,
                self.stir.log_blowup,
            );
        }
        self.input_mmcs.commit(ldes)
    }

    #[instrument(name = "STIR PCS open", skip_all)]
    fn open(
        &self,
        commitment_data_with_opening_points: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Step 1: Compute evaluations at opening points using Lagrange interpolation.
        let mats_and_points: Vec<_> = commitment_data_with_opening_points
            .iter()
            .map(|(data, points)| {
                let mats = self
                    .input_mmcs
                    .get_matrices(data)
                    .into_iter()
                    .map(|m| m.as_view())
                    .collect_vec();
                (mats, points)
            })
            .collect();

        let (global_max_height, global_max_width) = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats.iter().map(|m| (m.height(), m.width())))
            .reduce(|(hmax, wmax), (h, w)| (hmax.max(h), wmax.max(w)))
            .expect("No matrices supplied");
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Coset for the LDE: `GENERATOR * H` in bit-reversed order.
        let coset: Vec<Val> = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut pts = coset.iter().collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };

        let inv_denoms = compute_inverse_denominators::<Val, Challenge>(&mats_and_points, &coset);

        // Precompute adjusted barycentric weights once per opening point.
        // adjusted[i] = 1/(z - x_i) - 1/z, reused across all matrices opened at z.
        let adjusted_weights: LinearMap<Challenge, Vec<Challenge>> = inv_denoms
            .iter()
            .map(|(point, denoms)| (*point, compute_adjusted_weights(*point, denoms)))
            .collect();

        let all_opened_values: OpenedValues<Challenge> = mats_and_points
            .iter()
            .map(|(mats, points)| {
                izip!(mats.iter(), points.iter())
                    .map(|(mat, points_for_mat)| {
                        let h = mat.height() >> self.stir.log_blowup;
                        let (low_coset, _) = mat.split_rows(h);

                        points_for_mat
                            .iter()
                            .map(|&point| {
                                // Slice the precomputed adjusted weights to match this matrix's height.
                                // Zero-allocation hot path: straight to the SIMD dot product.
                                let adj = &adjusted_weights.get(&point).unwrap()[..h];
                                let ys = low_coset.interpolate_coset_with_precomputation(
                                    Val::GENERATOR,
                                    point,
                                    adj,
                                );
                                challenger.observe_algebra_slice(&ys);
                                ys
                            })
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec();

        // Step 2: Alpha-batch into a single reduced-opening vector.
        let alpha: Challenge = challenger.sample_algebra_element();
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers: Vec<Challenge> =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // `reduced[log_h]`: alpha-batched reduced-opening at log-height `log_h`. Keyed by
        // arbitrary `log_h` so the structure scales with the field's two-adicity rather than
        // a hardcoded array bound.
        let mut reduced_openings: alloc::collections::BTreeMap<usize, Vec<Challenge>> =
            alloc::collections::BTreeMap::new();
        let mut num_reduced: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();

        for ((mats, points), opened_vals) in mats_and_points.iter().zip(&all_opened_values) {
            for ((mat, points_for_mat), opened_for_mat) in
                izip!(mats.iter(), points.iter()).zip(opened_vals.iter())
            {
                let log_h = log2_strict_usize(mat.height());
                let ro = reduced_openings
                    .entry(log_h)
                    .or_insert_with(|| vec![Challenge::ZERO; mat.height()]);

                // Precompute alpha-batched row values for this matrix (reused per point).
                let p_x_vec: Vec<Challenge> = mat
                    .rows()
                    .map(|row| {
                        row.zip(alpha_powers.iter())
                            .map(|(px, &ap)| Challenge::from(px) * ap)
                            .sum()
                    })
                    .collect_vec();

                for (point, ys) in points_for_mat.iter().zip(opened_for_mat.iter()) {
                    let height_count = num_reduced.entry(log_h).or_insert(0);
                    let alpha_pow_offset = alpha.exp_u64(*height_count as u64);
                    *height_count += ys.len();

                    let full_height = mat.height();
                    let inv_denom = &inv_denoms.get(point).unwrap()[..full_height];

                    let y_combined: Challenge = ys
                        .iter()
                        .zip(alpha_powers.iter())
                        .map(|(&y, &ap)| y * ap)
                        .sum();

                    for (ro_val, (&inv_d, &p_x)) in
                        ro.iter_mut().zip(inv_denom.iter().zip(p_x_vec.iter()))
                    {
                        *ro_val += alpha_pow_offset * (p_x - y_combined) * inv_d;
                    }
                }
            }
        }

        // Step 3: For each non-empty height bucket (descending), run STIR on the bucket's
        // reduced opening and bind the input MMCS. Each distinct LDE height gets its own
        // STIR sub-proof. BTreeMap iterates in ascending key order, so reverse for descending.
        let mut bucket_proofs = Vec::new();

        let bucket_log_heights: Vec<usize> = reduced_openings.keys().rev().copied().collect();
        for log_h in bucket_log_heights {
            let ro = reduced_openings
                .remove(&log_h)
                .expect("present by construction");
            let bucket_height = 1usize << log_h;

            let mut ro_natural = ro;
            reverse_slice_index_bits(&mut ro_natural);
            let stir_coeffs = coeffs_from_codeword(&self.dft, &ro_natural, Val::GENERATOR);

            let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
            let stir_config = StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                log_stir_degree,
                self.stir.clone(),
            );

            let (stir_proof, first_round_query_indices) =
                prove_stir(&stir_config, stir_coeffs, &self.dft, challenger);

            // Input binding for this bucket. Folding factor is constant across rounds.
            let log_arity0 = stir_config.log_folding_factor;
            let fold_height0 = (1usize << stir_config.log_starting_domain_size()) >> log_arity0;
            let arity0 = 1usize << log_arity0;

            let input_openings: Vec<Vec<BatchOpening<Val, InputMmcs>>> =
                commitment_data_with_opening_points
                    .iter()
                    .map(|(data, _)| {
                        let mats = self.input_mmcs.get_matrices(data);
                        if !mats.iter().any(|m| m.height() == bucket_height) {
                            return Vec::new();
                        }
                        let commit_max_height = mats.iter().map(|m| m.height()).max().unwrap();
                        let log_commit_max = log2_strict_usize(commit_max_height);

                        first_round_query_indices
                            .iter()
                            .flat_map(|&j| {
                                (0..arity0).map(move |l| {
                                    let p = j + l * fold_height0;
                                    let q_local = reverse_bits_len(p, log_h);
                                    let q_global = q_local << (log_commit_max - log_h);
                                    self.input_mmcs.open_batch(q_global, data)
                                })
                            })
                            .collect()
                    })
                    .collect();

            bucket_proofs.push((stir_proof, input_openings));
        }

        (all_opened_values, bucket_proofs)
    }

    fn verify(
        &self,
        commitments_with_opening_points: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let global_max_height = commitments_with_opening_points
            .iter()
            .flat_map(|(_, domain_claims)| {
                domain_claims
                    .iter()
                    .map(|(domain, _)| domain.size() << self.stir.log_blowup)
            })
            .max()
            .unwrap_or(1);
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Observe all opened values to keep the transcript in sync.
        for (_, domain_claims) in &commitments_with_opening_points {
            for (_, point_claims) in domain_claims {
                for (_, opened_vals) in point_claims {
                    challenger.observe_algebra_slice(opened_vals);
                }
            }
        }

        let alpha: Challenge = challenger.sample_algebra_element();

        // Determine the set of distinct LDE-height buckets (descending) from the public domains.
        // Must match the prover's bucket iteration order.
        let bucket_log_heights: Vec<usize> = {
            let mut seen = BTreeSet::new();
            for (_, domain_claims) in &commitments_with_opening_points {
                for (domain, _) in domain_claims {
                    seen.insert(log2_strict_usize(domain.size() << self.stir.log_blowup));
                }
            }
            seen.into_iter().rev().collect()
        };

        if proof.len() != bucket_log_heights.len() {
            return Err(StirError::InvalidProofShape);
        }

        // Precompute alpha_pow_offset for each (commit, mat, point) triple. Tracked per
        // log_h via a BTreeMap so the structure scales with the field's two-adicity rather
        // than a hardcoded array length.
        let mut height_num_reduced: alloc::collections::BTreeMap<usize, usize> =
            alloc::collections::BTreeMap::new();
        let alpha_offsets: Vec<Vec<Vec<Challenge>>> = commitments_with_opening_points
            .iter()
            .map(|(_, domain_claims)| {
                domain_claims
                    .iter()
                    .map(|(domain, point_claims)| {
                        let log_h = log2_strict_usize(domain.size() << self.stir.log_blowup);
                        point_claims
                            .iter()
                            .map(|(_, vals)| {
                                let height_count = height_num_reduced.entry(log_h).or_insert(0);
                                let offset = alpha.exp_u64(*height_count as u64);
                                *height_count += vals.len();
                                offset
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let global_max_width = commitments_with_opening_points
            .iter()
            .flat_map(|(_, domain_claims)| {
                domain_claims
                    .iter()
                    .flat_map(|(_, point_claims)| point_claims.iter().map(|(_, v)| v.len()))
            })
            .max()
            .unwrap_or(0);
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers: Vec<Challenge> =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        let coset: Vec<Val> = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut pts = coset.iter().collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };

        // Verify each height bucket's STIR sub-proof and input binding.
        for (bucket_idx, &log_h) in bucket_log_heights.iter().enumerate() {
            let bucket_height = 1usize << log_h;
            let (stir_proof, input_openings) = &proof[bucket_idx];

            let log_stir_degree = log_h.saturating_sub(self.stir.log_blowup).max(1);
            let stir_config = StirConfig::<Val, Challenge, StirMmcs, Challenger>::new(
                log_stir_degree,
                self.stir.clone(),
            );

            // Run STIR verification and consume its first-round query view directly. No
            // hand-mirrored transcript replay: anything that touches the FS state is contained
            // inside `verify_stir` itself.
            let stir_outputs = verify_stir(&stir_config, stir_proof, challenger).map_err(|e| {
                e.map_input_err(|_| unreachable!("verify_stir does not produce InputError"))
            })?;

            // The folding factor is constant across rounds, so the same arity/fold_height
            // applies whether STIR ran with intermediate rounds or only a final round.
            let log_arity0 = stir_config.log_folding_factor;
            let fold_height0 = (1usize << stir_config.log_starting_domain_size()) >> log_arity0;
            let arity0 = 1usize << log_arity0;

            let first_round_unique_js = stir_outputs.first_round_indices;
            let fiber_evals_per_query = stir_outputs.first_round_fiber_evals;

            // Verify input MMCS openings and check consistency with STIR fiber evals.
            for (commit_idx, ((commitment, domain_claims), per_commit_openings)) in
                commitments_with_opening_points
                    .iter()
                    .zip(input_openings.iter())
                    .enumerate()
            {
                let mat_lde_heights: Vec<usize> = domain_claims
                    .iter()
                    .map(|(domain, _)| domain.size() << self.stir.log_blowup)
                    .collect();

                if !mat_lde_heights.contains(&bucket_height) {
                    if !per_commit_openings.is_empty() {
                        return Err(StirError::InvalidProofShape);
                    }
                    continue;
                }

                let commit_max_height = mat_lde_heights.iter().copied().max().unwrap();
                let log_commit_max = log2_strict_usize(commit_max_height);

                let mat_widths: Vec<usize> = domain_claims
                    .iter()
                    .map(|(_, point_claims)| {
                        point_claims.first().map(|(_, v)| v.len()).unwrap_or(0)
                    })
                    .collect();

                let dimensions: Vec<p3_matrix::Dimensions> = mat_lde_heights
                    .iter()
                    .zip(mat_widths.iter())
                    .map(|(&h, &w)| p3_matrix::Dimensions {
                        height: h,
                        width: w,
                    })
                    .collect();

                let mut opening_idx = 0usize;

                for (q_idx, &j) in first_round_unique_js.iter().enumerate() {
                    let row_evals = &fiber_evals_per_query[q_idx];
                    if row_evals.len() != arity0 {
                        return Err(StirError::InvalidProofShape);
                    }
                    #[allow(clippy::needless_range_loop)]
                    for l in 0..arity0 {
                        let p = j + l * fold_height0;
                        let q_local = reverse_bits_len(p, log_h);
                        let q_global = q_local << (log_commit_max - log_h);

                        let batch_open = &per_commit_openings[opening_idx];
                        opening_idx += 1;

                        let batch_ref = BatchOpeningRef::new(
                            &batch_open.opened_values,
                            &batch_open.opening_proof,
                        );
                        self.input_mmcs
                            .verify_batch(commitment, &dimensions, q_global, batch_ref)
                            .map_err(StirError::InputError)?;

                        let mut expected_ro = Challenge::ZERO;

                        for (mat_idx, (_, point_claims)) in domain_claims.iter().enumerate() {
                            if mat_lde_heights[mat_idx] != bucket_height {
                                continue;
                            }

                            let row_vals = &batch_open.opened_values[mat_idx];
                            let p_x: Challenge = row_vals
                                .iter()
                                .zip(alpha_powers.iter())
                                .map(|(&v, &ap)| Challenge::from(v) * ap)
                                .sum();

                            for (point_idx, (point, vals)) in point_claims.iter().enumerate() {
                                let alpha_pow_offset =
                                    alpha_offsets[commit_idx][mat_idx][point_idx];

                                let y_combined: Challenge = vals
                                    .iter()
                                    .zip(alpha_powers.iter())
                                    .map(|(&y, &ap)| y * ap)
                                    .sum();

                                let inv_denom =
                                    (*point - Challenge::from(coset[q_local])).inverse();

                                expected_ro += alpha_pow_offset * (p_x - y_combined) * inv_denom;
                            }
                        }

                        if expected_ro != row_evals[l] {
                            return Err(StirError::InvalidProofShape);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

type MatricesAndPoints<'a, F, EF> = (Vec<RowMajorMatrixView<'a, F>>, &'a Vec<Vec<EF>>);

/// Compute `1/(z - x)` for all coset elements `x`, batched over all unique points `z`.
fn compute_inverse_denominators<'a, F: TwoAdicField, EF: ExtensionField<F>>(
    mats_and_points: &'a [MatricesAndPoints<'a, F, EF>],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    // Find the maximum height for each unique opening point.
    let mut point_max_height: LinearMap<EF, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in mats.iter().zip(points.iter()) {
            for &point in points_for_mat {
                if let Some(existing) = point_max_height.get_mut(&point) {
                    if mat.height() > *existing {
                        *existing = mat.height();
                    }
                } else {
                    point_max_height.insert(point, mat.height());
                }
            }
        }
    }

    point_max_height
        .into_iter()
        .map(|(z, max_h)| {
            let max_h = max_h.max(1);
            let diffs: Vec<EF> = coset[..max_h].iter().map(|&x| z - EF::from(x)).collect();
            let inv_diffs = batch_multiplicative_inverse(&diffs);
            (z, inv_diffs)
        })
        .collect()
}
