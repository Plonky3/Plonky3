use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;
use p3_fri::verifier::FriError;
use p3_fri::{CodeFamily, Codeword, FriConfig, FriProof};

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs, PolynomialSpace};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, dot_product, ExtensionField,
    Field, TwoAdicField,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::{BitReversableMatrix, BitReversalPerm};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits, VecExt};
use serde::{Deserialize, Serialize};
use tracing::{info_span, instrument};

use crate::deep_quotient::deep_reduce_matrices;
use crate::{MultiplicativeCoset, RsCode};

/*
use crate::verifier::{self, FriError};
use crate::{prover, FriConfig, FriGenericConfig, FriProof};
*/

#[derive(Debug)]
pub struct MultplicativeFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    log_blowup: usize,
    dft: Dft,
    mmcs: InputMmcs,
    fri: FriConfig<FriMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> MultplicativeFriPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub const fn new(
        log_blowup: usize,
        dft: Dft,
        mmcs: InputMmcs,
        fri: FriConfig<FriMmcs>,
    ) -> Self {
        Self {
            log_blowup,
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(bound = "")]
pub struct BatchOpening<Val: Field, InputMmcs: Mmcs<Val>> {
    pub opened_values: Vec<Vec<Val>>,
    pub opening_proof: <InputMmcs as Mmcs<Val>>::Proof,
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for MultplicativeFriPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: TwoAdicField + ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = MultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;

    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error>;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        let log_n = log2_strict_usize(degree);
        MultiplicativeCoset {
            log_n,
            shift: Val::one(),
        }
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let ldes: Vec<_> = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                assert_eq!(domain.size(), evals.height());
                let shift = Val::generator() / domain.shift;
                // Commit to the bit-reversed LDE.
                self.dft
                    .coset_lde_batch(evals, self.log_blowup, shift)
                    .bit_reverse_rows()
                    .to_row_major_matrix()
            })
            .collect();

        self.mmcs.commit(ldes)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        // todo: handle extrapolation for LDEs we don't have
        assert_eq!(domain.shift, Val::generator());
        let lde = self.mmcs.get_matrices(prover_data)[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Batch combination challenge
        let alpha: Challenge = challenger.sample_ext_element();

        // log_height -> (alpha offset, reduced openings column)
        let mut reduced_openings: BTreeMap<usize, (Challenge, Vec<Challenge>)> = BTreeMap::new();

        let opened_values: OpenedValues<Challenge> = rounds
            .iter()
            .map(|(data, points_for_mats)| {
                let mats = self
                    .mmcs
                    .get_matrices(data)
                    .into_iter()
                    .map(|m| m.as_view())
                    .collect_vec();

                izip!(mats, points_for_mats)
                    .map(|(mat, points_for_mat)| {
                        let log_height = log2_strict_usize(mat.height());
                        let (alpha_offset, reduced_opening_for_log_height) =
                            reduced_openings.entry(log_height).or_insert_with(|| {
                                (Challenge::one(), vec![Challenge::zero(); 1 << log_height])
                            });

                        points_for_mat
                            .iter()
                            .map(|&zeta| {
                                let (low_coset, _) =
                                    mat.split_rows(mat.height() >> self.log_blowup);
                                let ps_at_zeta =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            interpolate_coset(
                                                &BitReversalPerm::new_view(low_coset),
                                                Val::generator(),
                                                zeta,
                                            )
                                        });

                                let reduced_ps_at_zeta: Challenge =
                                    dot_product(alpha.powers(), ps_at_zeta.iter().copied());

                                let mut subgroup = cyclic_subgroup_coset_known_order(
                                    Val::two_adic_generator(log_height),
                                    Val::generator(),
                                    1 << log_height,
                                )
                                .collect_vec();
                                reverse_slice_index_bits(&mut subgroup);
                                let inv_denoms = batch_multiplicative_inverse(&subgroup);

                                info_span!("reduce rows").in_scope(|| {
                                    mat.dot_ext_powers(alpha)
                                        .zip(reduced_opening_for_log_height.par_iter_mut())
                                        .zip(inv_denoms.par_iter())
                                        .for_each(|((reduced_row, ro), &inv_denom)| {
                                            *ro += *alpha_offset
                                                * (reduced_row - reduced_ps_at_zeta)
                                                * inv_denom
                                        })
                                });

                                *alpha_offset *= alpha.exp_u64(mat.width() as u64);

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let log_global_max_height = *reduced_openings.last_entry().unwrap().key();

        let codewords = reduced_openings
            .into_iter()
            .map(|(log_height, (_, word))| {
                Codeword::full(
                    RsCode::new(
                        self.log_blowup,
                        log_height - self.log_blowup,
                        Challenge::from_base(Val::generator()),
                    ),
                    word,
                )
            })
            .collect();

        let proof = p3_fri::prover::prove(&self.fri, codewords, challenger, |index| {
            rounds
                .iter()
                .map(|(data, _)| {
                    let log_max_height = log2_strict_usize(self.mmcs.get_max_height(data));
                    let bits_reduced = log_global_max_height - log_max_height;
                    let reduced_index = index >> bits_reduced;
                    let (opened_values, opening_proof) = self.mmcs.open_batch(reduced_index, data);
                    BatchOpening {
                        opened_values,
                        opening_proof,
                    }
                })
                .collect_vec()
        });

        (opened_values, proof)
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let codes: Vec<RsCode<Challenge>> = rounds
            .iter()
            .flat_map(|(_, mats)| {
                mats.iter()
                    .map(|(domain, _)| log2_strict_usize(domain.size()))
            })
            .sorted()
            .map(|log_message_len| {
                RsCode::new(self.log_blowup, log_message_len, Val::generator().into())
            })
            .collect();

        let log_max_word_len = codes.iter().map(|c| c.log_word_len()).max().unwrap();

        // Batch combination challenge
        let alpha: Challenge = challenger.sample_ext_element();

        p3_fri::verifier::verify(
            &self.fri,
            &codes,
            proof,
            challenger,
            |index, input_proof| {
                // log_height -> (alpha_pow, reduced_opening)
                let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

                for (batch_opening, (_batch_commit, mats)) in izip!(input_proof, &rounds) {
                    // todo: check input mmcs

                    for (mat_opening, (mat_domain, mat_points_and_values)) in
                        izip!(&batch_opening.opened_values, mats)
                    {
                        let log_height = log2_strict_usize(mat_domain.size()) + self.log_blowup;

                        let bits_reduced = log_max_word_len - log_height;
                        let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

                        // todo: this can be nicer with domain methods?

                        let x = Val::generator()
                            * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                        let (alpha_pow, ro) = reduced_openings
                            .entry(log_height)
                            .or_insert((Challenge::one(), Challenge::zero()));

                        for (z, ps_at_z) in mat_points_and_values {
                            for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                                let quotient = (-p_at_z + p_at_x) / (-*z + x);
                                *ro += *alpha_pow * quotient;
                                *alpha_pow *= alpha;
                            }
                        }
                    }
                }

                reduced_openings.into_values().map(|(_, ro)| ro).collect()
            },
        )?;

        // let log_global_max_height = proof.commit_phase_commits.len() + self.fri.log_blowup;
        /*



        let g: TwoAdicFriGenericConfigForMmcs<Val, InputMmcs> =
            TwoAdicFriGenericConfig(PhantomData);

        verifier::verify(&g, &self.fri, proof, challenger, |index, input_proof| {
            // TODO: separate this out into functions

            // log_height -> (alpha_pow, reduced_opening)
            let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

            for (batch_opening, (batch_commit, mats)) in izip!(input_proof, &rounds) {
                let batch_heights = mats
                    .iter()
                    .map(|(domain, _)| domain.size() << self.fri.log_blowup)
                    .collect_vec();
                let batch_dims = batch_heights
                    .iter()
                    // TODO: MMCS doesn't really need width; we put 0 for now.
                    .map(|&height| Dimensions { width: 0, height })
                    .collect_vec();

                let batch_max_height = batch_heights.iter().max().expect("Empty batch?");
                let log_batch_max_height = log2_strict_usize(*batch_max_height);
                let bits_reduced = log_global_max_height - log_batch_max_height;
                let reduced_index = index >> bits_reduced;

                self.mmcs.verify_batch(
                    batch_commit,
                    &batch_dims,
                    reduced_index,
                    &batch_opening.opened_values,
                    &batch_opening.opening_proof,
                )?;
                for (mat_opening, (mat_domain, mat_points_and_values)) in
                    izip!(&batch_opening.opened_values, mats)
                {
                    let log_height = log2_strict_usize(mat_domain.size()) + self.fri.log_blowup;

                    let bits_reduced = log_global_max_height - log_height;
                    let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

                    // todo: this can be nicer with domain methods?

                    let x = Val::generator()
                        * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

                    let (alpha_pow, ro) = reduced_openings
                        .entry(log_height)
                        .or_insert((Challenge::one(), Challenge::zero()));

                    for (z, ps_at_z) in mat_points_and_values {
                        for (&p_at_x, &p_at_z) in izip!(mat_opening, ps_at_z) {
                            let quotient = (-p_at_z + p_at_x) / (-*z + x);
                            *ro += *alpha_pow * quotient;
                            *alpha_pow *= alpha;
                        }
                    }
                }
            }

            // Return reduced openings descending by log_height.
            Ok(reduced_openings
                .into_iter()
                .rev()
                .map(|(log_height, (_alpha_pow, ro))| (log_height, ro))
                .collect())
        })
        .expect("fri err");
        */

        Ok(())
    }
}
