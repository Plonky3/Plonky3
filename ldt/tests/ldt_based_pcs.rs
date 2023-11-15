use std::marker::PhantomData;

use itertools::izip;
use p3_baby_bear::BabyBear;
use p3_blake3::Blake3;
use p3_challenger::{DuplexChallenger, FieldChallenger};
use p3_commit::{DirectMmcs, ExtensionMmcs, Mmcs, OpenedValues, Pcs, UnivariatePcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, ExtensionField, Field, TwoAdicField};
use p3_ldt::{Ldt, LdtBasedPcs, QuotientMmcs};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Dimensions, Matrix, MatrixGet, MatrixRows};
use p3_mds::coset_mds::CosetMds;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
use rand::{thread_rng, Rng};

// trivial LDT, assumes committing to the same values produces the same commitment
struct TrivialLdt;

#[derive(Debug, PartialEq, Eq)]
enum TrivialLdtErr<MmcsErr> {
    HighDegree,
    MmcsErr(MmcsErr),
}

impl<Val, M, Challenger> Ldt<Val, M, Challenger> for TrivialLdt
where
    Val: Field + TwoAdicField,
    M: Mmcs<Val>,
    Challenger: FieldChallenger<Val>,
{
    // for each batch, for each row, (matrix, row, proof)
    type Proof = Vec<Vec<(Vec<Vec<Val>>, M::Proof)>>;
    type Error = TrivialLdtErr<M::Error>;

    fn log_blowup(&self) -> usize {
        1
    }

    fn prove(
        &self,
        input_mmcs: &[M],
        input_data: &[&M::ProverData],
        _challenger: &mut Challenger,
    ) -> Self::Proof {
        input_mmcs
            .iter()
            .zip(input_data)
            .map(|(mmcs, data)| {
                let max_height = mmcs.get_max_height(data);

                for mat in mmcs.get_matrices(data) {
                    let mat = mat.to_row_major_matrix();
                    dbg!(mat.dimensions());
                    // dbg!(&mat.values);
                    let poly = Radix2Dit.idft_batch(mat.to_row_major_matrix());
                    // dbg!(&poly.values);
                    if poly.last_row().into_iter().any(|x| !x.is_zero()) {
                        panic!("TrivialLdt tried to commit to a high-degree matrix");
                    }
                }

                (0..max_height).map(|i| mmcs.open_batch(i, data)).collect()
            })
            .collect()
    }

    fn verify(
        &self,
        input_mmcs: &[M],
        input_commits: &[M::Commitment],
        proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        for (mmcs, comm, proof) in izip!(input_mmcs, input_commits, proof) {
            let n_mats = proof[0].0.len();
            let mut mats: Vec<Vec<Val>> = vec![vec![]; n_mats];
            let dims: Vec<_> = proof[0]
                .0
                .iter()
                .map(|m| Dimensions {
                    width: m.len(),
                    height: proof.len(),
                })
                .collect();
            for (r, row) in proof.iter().enumerate() {
                if let Err(e) = mmcs.verify_batch(comm, &dims, r, &row.0, &row.1) {
                    return Err(TrivialLdtErr::MmcsErr(e));
                }
                for (i, values) in row.0.iter().enumerate() {
                    mats[i].extend(values);
                }
            }
            let mats: Vec<RowMajorMatrix<Val>> = mats
                .into_iter()
                .zip(dims)
                .map(|(m, d)| RowMajorMatrix::new(m, d.width))
                .collect();
            for mat in mats {
                let poly = Radix2Dit.idft_batch(mat.clone());
                if poly.last_row().into_iter().any(|x| !x.is_zero()) {
                    return Err(TrivialLdtErr::HighDegree);
                }
            }
        }
        Ok(())
    }
}

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type MyMds = CosetMds<F, 16>;
type Perm = Poseidon2<F, MyMds, DiffusionMatrixBabybear, 16, 5>;
type MyHash = SerializingHasher32<Blake3>;
type MyCompress = CompressionFunctionFromHasher<F, MyHash, 2, 8>;
type ValMmcs = FieldMerkleTreeMmcs<F, MyHash, MyCompress, 8>;
type Challenger = DuplexChallenger<F, Perm, 16>;
type Dft = Radix2Dit;

fn get_test_params() -> (ValMmcs, Challenger) {
    let mds = MyMds::default();
    let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
    let hash = MyHash::new(Blake3 {});
    let compress = MyCompress::new(hash);
    let val_mmcs = ValMmcs::new(hash, compress);
    (val_mmcs.clone(), Challenger::new(perm))
}

#[test]
#[ignore]
fn test_trivial_ldt() {
    let (mmcs, challenger) = get_test_params();

    let ldt = TrivialLdt;

    {
        let trace = RowMajorMatrix::rand_nonzero(&mut thread_rng(), 32, 10);
        let lde = Radix2Dit.coset_lde_batch(trace, 1, F::generator());

        let mut ch = challenger.clone();
        let (comm, data) = mmcs.commit(vec![lde]);
        let proof = ldt.prove(&[mmcs.clone()], &[&data], &mut ch);

        let mut ch = challenger.clone();
        assert_eq!(
            ldt.verify(&[mmcs.clone()], &[comm], &proof, &mut ch),
            Ok(())
        );
    }

    if false {
        let trace = RowMajorMatrix::rand_nonzero(&mut thread_rng(), 32, 10);

        let mut ch = challenger.clone();
        let (comm, data) = mmcs.commit(vec![trace]);
        let proof = ldt.prove(&[mmcs.clone()], &[&data], &mut ch);

        let mut ch = challenger.clone();
        assert_eq!(
            ldt.verify(&[mmcs.clone()], &[comm], &proof, &mut ch),
            Err(TrivialLdtErr::HighDegree)
        );
    }
}

type MyPcs = LdtBasedPcs<F, EF, Dft, ValMmcs, TrivialLdt, Challenger>;

#[test]
fn test_ldt_based_pcs() {
    let (mmcs, challenger) = get_test_params();
    let ldt = TrivialLdt;
    let pcs = MyPcs::new(Radix2Dit, mmcs.clone(), ldt);

    let poly = RowMajorMatrix::rand_nonzero(&mut thread_rng(), 8, 4);
    let trace = Radix2Dit.dft_batch(poly.clone());
    let (comm, data) = pcs.commit_batches(vec![trace]);

    let alpha: EF = thread_rng().gen();
    let poly_at_alpha: Vec<EF> = (0..poly.width())
        .map(|c| {
            let mut acc = EF::zero();
            for r in (0..poly.height()).rev() {
                acc *= alpha;
                acc += poly.get(r, c);
            }
            acc
        })
        .collect();

    let mut ch = challenger.clone();
    let (mut opening, proof) =
        <MyPcs as UnivariatePcs<F, EF, RowMajorMatrix<F>, Challenger>>::open_multi_batches(
            &pcs,
            &[(&data, &[alpha])],
            &mut ch,
        );

    // check the opening is actually correct
    assert_eq!(opening[0][0][0], poly_at_alpha);

    // check it verifies
    let mut ch = challenger.clone();
    assert_eq!(
        <MyPcs as UnivariatePcs<F, EF, RowMajorMatrix<F>, Challenger>>::verify_multi_batches(
            &pcs,
            &[(comm, &[alpha])],
            opening.clone(),
            &proof,
            &mut ch,
        ),
        Ok(())
    );

    // change opening so it's incorrect

    /*
    let err: EF = thread_rng().gen();
    opening[0][0][0][5] += err;

    assert_eq!(
        <MyPcs as UnivariatePcs<F, EF, RowMajorMatrix<F>, Challenger>>::verify_multi_batches(
            &pcs,
            &[(comm, &[alpha])],
            opening,
            &proof,
            &mut ch,
        ),
        Err(TrivialLdtErr::MmcsErr(()))
    );
    */
}
