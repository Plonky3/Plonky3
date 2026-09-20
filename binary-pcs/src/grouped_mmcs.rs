//! Experimental adjacent-symbol grouping for the binary PCS's single-column codewords.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::{Field, PackedValue};
use p3_matrix::{Dimensions, Matrix};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Groups adjacent codeword symbols into wider Merkle leaves without changing symbol queries.
///
/// This experimental adapter supports exactly one width-1, power-of-two-height matrix per
/// commitment, as used by [`crate::BinaryPcs`]. It is not a general matrix-batching adapter.
/// The verifier must configure the same group size independently of the proof. The group size
/// is capped at the codeword length in small rounds. Grouping changes roots and proof encoding,
/// but retains every symbol, every sampled fold pair, and every intermediate commitment.
#[derive(Clone, Debug)]
pub struct GroupedCodewordMmcs<Inner> {
    inner: Inner,
    group_size: usize,
    log_inv_rate: Option<usize>,
}

impl<Inner> GroupedCodewordMmcs<Inner> {
    /// Builds an adapter with a power-of-two number of symbols per leaf.
    ///
    /// # Panics
    /// Panics if `group_size` is zero or not a power of two.
    pub fn new(inner: Inner, group_size: usize) -> Self {
        assert!(
            group_size.is_power_of_two(),
            "group size must be a power of two"
        );
        Self {
            inner,
            group_size,
            log_inv_rate: None,
        }
    }

    /// Pack one full coset for the next fold batch, including a shorter final batch.
    /// Prover and verifier must use the same validated PCS configuration.
    pub const fn for_folding(inner: Inner, config: &crate::BinaryPcsConfig) -> Self {
        Self {
            inner,
            group_size: 1 << config.log_folding_factor(),
            log_inv_rate: Some(config.log_inv_rate()),
        }
    }

    fn group_size_at(&self, height: usize) -> Option<usize> {
        let message_len = height >> self.log_inv_rate.unwrap_or(0);
        (message_len != 0).then(|| self.group_size.min(message_len))
    }
}

/// A zero-copy view of adjacent rows as one row, retaining the original codeword for folding.
#[derive(Clone, Debug)]
pub struct GroupedCodeword<M> {
    matrix: M,
    group_size: usize,
}

impl<F: Field, M: Matrix<F>> Matrix<F> for GroupedCodeword<M> {
    fn width(&self) -> usize {
        self.group_size
    }

    fn height(&self) -> usize {
        self.matrix.height() / self.group_size
    }

    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
        (start..end).map(move |lane| {
            // SAFETY: the constructor admits only width-1 matrices whose height is divisible
            // by group_size. The caller guarantees r < height() and lane < group_size.
            unsafe { self.matrix.get_unchecked(r * self.group_size + lane, 0) }
        })
    }

    #[inline]
    fn vertically_packed_row<P>(&self, r: usize) -> impl Iterator<Item = P>
    where
        F: Copy,
        P: PackedValue<Value = F>,
    {
        let matrix = &self.matrix;
        let group_size = self.group_size;
        let height = self.height();
        let row = r % height;
        let no_wrap = P::WIDTH != 1 && r + P::WIDTH <= height;

        // Every read below is in bounds: the constructor admits only width-1 matrices whose
        // height is `height() * group_size`, each row index is below `height()`, and each
        // lane is below `group_size`.
        (0..group_size).map(move |lane| {
            if P::WIDTH == 1 {
                // SAFETY: `row < height()` and `lane < group_size`.
                unsafe { P::broadcast(matrix.get_unchecked(row * group_size + lane, 0)) }
            } else if no_wrap {
                // SAFETY: `r + i < height()` for every `i < P::WIDTH`, and `lane < group_size`.
                P::from_fn(|i| unsafe { matrix.get_unchecked((r + i) * group_size + lane, 0) })
            } else {
                P::from_fn(|i| {
                    let row = (r + i) % height;
                    // SAFETY: `row < height()` and `lane < group_size`.
                    unsafe { matrix.get_unchecked(row * group_size + lane, 0) }
                })
            }
        })
    }
}

/// Authentication of the grouped leaves plus symbols not already in the requested rows.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, P: Serialize",
    deserialize = "F: Deserialize<'de>, P: Deserialize<'de>"
))]
pub struct GroupedCodewordProof<F, P> {
    inner: P,
    /// In ascending symbol-index order across the distinct opened groups.
    missing_symbols: Vec<F>,
}

/// A malformed grouped opening or a failure of the underlying commitment check.
#[derive(Debug, Error)]
pub enum GroupedCodewordError<E> {
    /// This adapter accepts only one nonempty power-of-two-height column.
    #[error("expected one width-1 codeword of power-of-two length")]
    Dimensions,
    /// Indices, requested rows, or the number of supplemental symbols disagree.
    #[error("malformed grouped codeword opening")]
    OpeningShape,
    /// Repeated requests for the same symbol carry different values.
    #[error("conflicting duplicate symbol openings")]
    ConflictingDuplicate,
    /// The grouped rows failed authentication.
    #[error("grouped codeword authentication failed: {0:?}")]
    Inner(#[source] E),
}

fn group_indices(indices: &[usize], group_size: usize) -> Vec<usize> {
    let mut groups: Vec<_> = indices.iter().map(|i| i / group_size).collect();
    groups.sort_unstable();
    groups.dedup();
    groups
}

impl<F: Field, Inner: Mmcs<F>> Mmcs<F> for GroupedCodewordMmcs<Inner> {
    type ProverData<M> = Inner::ProverData<GroupedCodeword<M>>;
    type Commitment = Inner::Commitment;
    type Proof = GroupedCodewordProof<F, Inner::MultiProof>;
    type MultiProof = GroupedCodewordProof<F, Inner::MultiProof>;
    type Error = GroupedCodewordError<Inner::Error>;

    fn commit<M: Matrix<F>>(&self, mut inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        assert_eq!(inputs.len(), 1, "expected exactly one codeword");
        let matrix = inputs.pop().unwrap();
        assert_eq!(matrix.width(), 1, "expected a width-1 codeword");
        assert!(
            matrix.height().is_power_of_two(),
            "expected power-of-two length"
        );
        let group_size = self
            .group_size_at(matrix.height())
            .expect("codeword shorter than rate expansion");
        self.inner
            .commit_matrix(GroupedCodeword { matrix, group_size })
    }

    fn get_matrices<'a, M: Matrix<F>>(&self, data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner
            .get_matrices(data)
            .into_iter()
            .map(|view| &view.matrix)
            .collect()
    }

    fn open_batch<M: Matrix<F>>(
        &self,
        index: usize,
        data: &Self::ProverData<M>,
    ) -> BatchOpening<F, Self> {
        let (mut rows, proof) = self.open_multi_batch(&[index], data);
        BatchOpening::new(rows.pop().unwrap(), proof)
    }

    fn verify_batch(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opening: BatchOpeningRef<'_, F, Self>,
    ) -> Result<(), Self::Error> {
        let rows = vec![opening.opened_values.iter().map(Vec::as_slice).collect()];
        self.verify_multi_batch(
            commitment,
            dimensions,
            &[index],
            &rows,
            opening.opening_proof,
        )
    }

    fn open_multi_batch<M: Matrix<F>>(
        &self,
        indices: &[usize],
        data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<F>>>, Self::MultiProof) {
        let views = self.inner.get_matrices(data);
        let view = views[0];
        assert!(
            indices.iter().all(|&i| i < view.matrix.height()),
            "symbol index out of range"
        );
        let group_size = view.group_size;
        let groups = group_indices(indices, group_size);
        let (group_rows, inner) = self.inner.open_multi_batch(&groups, data);
        let requested: BTreeSet<_> = indices.iter().copied().collect();
        let mut missing_symbols = Vec::new();
        for (&group, rows) in groups.iter().zip(group_rows) {
            for (lane, &value) in rows[0].iter().enumerate() {
                if !requested.contains(&(group * group_size + lane)) {
                    missing_symbols.push(value);
                }
            }
        }
        let rows = indices
            .iter()
            .map(|&i| vec![vec![view.matrix.get(i, 0).unwrap()]])
            .collect();
        (
            rows,
            GroupedCodewordProof {
                inner,
                missing_symbols,
            },
        )
    }

    fn verify_multi_batch<R: AsRef<[F]> + PartialEq>(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        use GroupedCodewordError::{
            ConflictingDuplicate, Dimensions as BadDimensions, OpeningShape,
        };

        let [dim] = dimensions else {
            return Err(BadDimensions);
        };
        if dim.width != 1 || !dim.height.is_power_of_two() {
            return Err(BadDimensions);
        }
        let group_size = self.group_size_at(dim.height).ok_or(BadDimensions)?;
        if indices.len() != opened_values.len() {
            return Err(OpeningShape);
        }
        let mut requested = BTreeMap::new();
        for (&index, rows) in indices.iter().zip(opened_values) {
            if index >= dim.height || rows.len() != 1 || rows[0].as_ref().len() != 1 {
                return Err(OpeningShape);
            }
            let value = rows[0].as_ref()[0];
            if requested
                .insert(index, value)
                .is_some_and(|old| old != value)
            {
                return Err(ConflictingDuplicate);
            }
        }
        let groups = group_indices(indices, group_size);
        // Check before allocating complete grouped rows. All groups are disjoint and inside
        // the validated codeword, so the product is bounded by dim.height.
        let missing_count = groups.len() * group_size - requested.len();
        if proof.missing_symbols.len() != missing_count {
            return Err(OpeningShape);
        }
        let mut missing = proof.missing_symbols.iter();
        let rows: Vec<_> = groups
            .iter()
            .map(|&group| {
                let row: Vec<_> = (0..group_size)
                    .map(|lane| {
                        *requested
                            .get(&(group * group_size + lane))
                            .unwrap_or_else(|| {
                                missing.next().expect("missing symbol count checked above")
                            })
                    })
                    .collect();
                vec![row]
            })
            .collect();
        let grouped_dimensions = [Dimensions {
            width: group_size,
            height: dim.height / group_size,
        }];
        self.inner
            .verify_multi_batch(
                commitment,
                &grouped_dimensions,
                &groups,
                &rows,
                &proof.inner,
            )
            .map_err(GroupedCodewordError::Inner)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128 as F, TowerLevel};
    use p3_commit::{BatchOpeningRef, Mmcs};
    use p3_field::{FieldArray, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};

    use super::{GroupedCodeword, GroupedCodewordMmcs};
    use crate::test_util::mmcs;

    /// A width-3 packing, wide enough to reach both the no-wrap and the wrap-around reads.
    type Packed = FieldArray<F, 3>;

    /// Reuses `GroupedCodeword`'s indexing but leaves `vertically_packed_row` at the
    /// trait's default implementation, for comparison against the overridden version.
    struct DefaultPackedGroupedCodeword<M> {
        matrix: M,
        group_size: usize,
    }

    impl<M: Matrix<F>> Matrix<F> for DefaultPackedGroupedCodeword<M> {
        fn width(&self) -> usize {
            self.group_size
        }

        fn height(&self) -> usize {
            self.matrix.height() / self.group_size
        }

        unsafe fn row_subseq_unchecked(
            &self,
            r: usize,
            start: usize,
            end: usize,
        ) -> impl IntoIterator<Item = F, IntoIter = impl Iterator<Item = F> + Send + Sync> {
            (start..end).map(move |lane| {
                // SAFETY: same invariants as `GroupedCodeword::row_subseq_unchecked`.
                unsafe { self.matrix.get_unchecked(r * self.group_size + lane, 0) }
            })
        }
    }

    #[test]
    fn groups_match_an_explicitly_reshaped_commitment() {
        let values: Vec<_> = (0..32).map(F::from_repr).collect();
        for group_size in [1, 2, 4, 8, 16, 64] {
            let grouped = GroupedCodewordMmcs::new(mmcs(), group_size);
            let (root, data) = grouped.commit_matrix(RowMajorMatrix::new(values.clone(), 1));
            let (expected, _) =
                mmcs().commit_matrix(RowMajorMatrix::new(values.clone(), group_size.min(32)));
            assert_eq!(root, expected);
            assert_eq!(grouped.get_matrices(&data)[0].values, values);
            assert_eq!(grouped.get_matrices(&data)[0].width, 1);

            let indices = [31, 0, 1, 17, 0];
            let (rows, proof) = grouped.open_multi_batch(&indices, &data);
            for (&index, row) in indices.iter().zip(&rows) {
                assert_eq!(row, &vec![vec![values[index]]]);
            }
            let dims = [Dimensions {
                width: 1,
                height: 32,
            }];
            grouped
                .verify_multi_batch(&root, &dims, &indices, &rows, &proof)
                .unwrap();

            let opening = grouped.open_batch(17, &data);
            grouped
                .verify_batch(&root, &dims, 17, (&opening).into())
                .unwrap();
        }
    }

    #[test]
    fn pairs_need_no_extra_symbols_and_larger_groups_only_send_missing_symbols() {
        let values: Vec<_> = (0..16).map(F::from_repr).collect();
        let indices = [0, 1, 8, 9, 0, 1];
        for (size, missing) in [(2, vec![]), (4, vec![2, 3, 10, 11])] {
            let grouped = GroupedCodewordMmcs::new(mmcs(), size);
            let (_, data) = grouped.commit_matrix(RowMajorMatrix::new(values.clone(), 1));
            let (_, proof) = grouped.open_multi_batch(&indices, &data);
            assert_eq!(
                proof.missing_symbols,
                missing.into_iter().map(F::from_repr).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn folding_schedule_sizes_leaves_for_the_next_actual_batch() {
        use crate::{BinaryPcsConfig, BinaryPcsParams};
        let config = BinaryPcsConfig::try_new::<F, F>(
            6,
            BinaryPcsParams {
                log_inv_rate: 2,
                security_level: 100,
                pow_bits: 0,
            },
        )
        .unwrap()
        .try_with_folding(4)
        .unwrap();
        let grouped = GroupedCodewordMmcs::for_folding(mmcs(), &config);
        for (len, width) in [(256, 16), (16, 4)] {
            let values: Vec<_> = (0..len).map(|i| F::from_repr(i as u128)).collect();
            let (root, data) = grouped.commit_matrix(RowMajorMatrix::new(values.clone(), 1));
            let (expected, _) = mmcs().commit_matrix(RowMajorMatrix::new(values, width));
            assert_eq!(root, expected);
            let opening = grouped.open_batch(len - 1, &data);
            grouped
                .verify_batch(
                    &root,
                    &[Dimensions {
                        height: len,
                        width: 1,
                    }],
                    len - 1,
                    (&opening).into(),
                )
                .unwrap();
        }
    }

    #[test]
    fn malformed_openings_and_unauthenticated_group_members_are_rejected() {
        let grouped = GroupedCodewordMmcs::new(mmcs(), 4);
        let values: Vec<_> = (0..32).map(F::from_repr).collect();
        let (root, data) = grouped.commit_matrix(RowMajorMatrix::new(values, 1));
        let indices = [0, 1, 0];
        let dims = [Dimensions {
            width: 1,
            height: 32,
        }];
        let (rows, proof) = grouped.open_multi_batch(&indices, &data);
        let verify = |rows: &[Vec<Vec<F>>], proof: &_| {
            grouped.verify_multi_batch(&root, &dims, &indices, rows, proof)
        };
        assert!(verify(&rows, &proof).is_ok());
        for position in 0..proof.missing_symbols.len() {
            let mut bad = proof.clone();
            bad.missing_symbols[position] += F::ONE;
            assert!(verify(&rows, &bad).is_err());
        }
        let mut bad = proof.clone();
        bad.missing_symbols.pop();
        assert!(verify(&rows, &bad).is_err());
        let mut bad = proof.clone();
        bad.missing_symbols.push(F::ZERO);
        assert!(verify(&rows, &bad).is_err());
        let mut bad_rows = rows.clone();
        bad_rows[2][0][0] += F::ONE;
        assert!(verify(&bad_rows, &proof).is_err());
        assert!(verify(&rows[..2], &proof).is_err());
        let mut bad_rows = rows.clone();
        bad_rows[0][0].push(F::ZERO);
        assert!(verify(&bad_rows, &proof).is_err());
        let mut bad_rows = rows.clone();
        bad_rows[0].clear();
        assert!(verify(&bad_rows, &proof).is_err());
        assert!(
            grouped
                .verify_multi_batch(&root, &dims, &[0, 1, 32], &rows, &proof)
                .is_err()
        );
        for dims in [
            vec![],
            vec![Dimensions {
                width: 2,
                height: 16,
            }],
            vec![Dimensions {
                width: 1,
                height: 0,
            }],
            vec![Dimensions {
                width: 1,
                height: 31,
            }],
        ] {
            assert!(
                grouped
                    .verify_multi_batch(&root, &dims, &indices, &rows, &proof)
                    .is_err()
            );
        }
        // The verifier chooses its grouping; proof-supplied rows cannot override it.
        assert!(
            GroupedCodewordMmcs::new(mmcs(), 8)
                .verify_multi_batch(&root, &dims, &indices, &rows, &proof)
                .is_err()
        );
        assert!(
            grouped
                .verify_batch(&root, &dims, 0, BatchOpeningRef::new(&rows[0], &proof))
                .is_err()
        );
    }

    #[test]
    fn full_pcs_proofs_round_trip_with_grouped_leaves() {
        use p3_commit::MultilinearPcs;
        use p3_sumcheck::layout::{Layout, SuffixProver, Table};
        use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        use crate::test_util::{MyMmcs, challenger};
        use crate::{BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BinaryPcsProof};

        for num_variables in [1, 6, 10] {
            for group_size in [1, 2, 4, 8, 16] {
                let config = BinaryPcsConfig::try_new::<F, F>(
                    num_variables,
                    BinaryPcsParams {
                        log_inv_rate: 2,
                        security_level: 100,
                        pow_bits: 0,
                    },
                )
                .unwrap();
                let pcs = BinaryPcs::new(
                    config,
                    GroupedCodewordMmcs::new(mmcs(), group_size),
                    GroupedCodewordMmcs::new(mmcs(), group_size),
                )
                .unwrap();
                let mut rng = SmallRng::seed_from_u64(0x6710);
                let table = Table::rand(&mut rng, 1, num_variables);
                let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);
                let protocol = OpeningProtocol::new(vec![TableSpec::new(
                    TableShape::new(num_variables, 1),
                    vec![OpeningBatch::new(vec![0], Vec::new())],
                )]);
                let mut prover_challenger = challenger();
                let (commitment, data) = pcs.commit(witness, &mut prover_challenger).unwrap();
                let proof = pcs
                    .open(data, protocol.clone(), &mut prover_challenger)
                    .unwrap();
                let bytes = postcard::to_allocvec(&proof).unwrap();
                let decoded: BinaryPcsProof<
                    F,
                    F,
                    GroupedCodewordMmcs<MyMmcs>,
                    GroupedCodewordMmcs<MyMmcs>,
                > = postcard::from_bytes(&bytes).unwrap();
                pcs.verify(&commitment, &decoded, &mut challenger(), protocol)
                    .unwrap();
            }
        }
    }

    #[test]
    fn vertically_packed_row_matches_the_default_trait_implementation_on_wrap_around() {
        // Odd group sizes and lengths that don't divide the packing width, so both the
        // fast path and the wrap-around fallback get exercised.
        for (len, group_size) in [(24usize, 3usize), (30, 5), (14, 7), (2, 1), (9, 9)] {
            let values: Vec<_> = (0..len as u128).map(F::from_repr).collect();
            let grouped = GroupedCodeword {
                matrix: RowMajorMatrix::new(values.clone(), 1),
                group_size,
            };
            let reference = DefaultPackedGroupedCodeword {
                matrix: RowMajorMatrix::new(values, 1),
                group_size,
            };
            let height = grouped.height();
            assert_eq!(height, reference.height());

            for r in 0..height {
                // Width-1 packing exercises the scalar broadcast branch.
                let actual: Vec<F> = grouped.vertically_packed_row::<F>(r).collect();
                let expected: Vec<F> = reference.vertically_packed_row::<F>(r).collect();
                assert_eq!(
                    actual, expected,
                    "width 1, len {len}, group {group_size}, r {r}"
                );

                // Width-3 packing exercises both the no-wrap fast path and, once `r` gets
                // close to `height`, the modulo wrap-around path.
                let actual: Vec<Packed> = grouped.vertically_packed_row::<Packed>(r).collect();
                let expected: Vec<Packed> = reference.vertically_packed_row::<Packed>(r).collect();
                assert_eq!(
                    actual, expected,
                    "width 3, len {len}, group {group_size}, r {r}"
                );
            }
        }
    }
}
