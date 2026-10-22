//! A WHIR opening verified inside a circuit, whose own proof then verifies natively.

mod common;

use common::{Challenger, EF, F, ValMmcs, chip, config, prove_and_verify};
use p3_challenger::CanSampleUniformBits;
use p3_circuit::tables::CircuitAir;
use p3_circuit::{CircuitArith, Ext, MirrorChallenger};
use p3_commit::{MmcsArith, MultilinearPcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_koala_bear::default_koalabear_poseidon2_16;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout, SuffixProver, Table, observe_commitment};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_verifier_arith::Lift;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::{PcsProof, WhirProver};

/// The polynomial's number of variables.
const NUM_VARIABLES: usize = 10;

/// Variables each WHIR round folds.
const FOLDING: usize = 4;

/// The circuit arithmetic the opening is verified in.
type Circuit = CircuitArith<F, EF, common::Chip>;

/// A commitment of the scheme.
type Commitment = <ValMmcs as p3_commit::Mmcs<F>>::Commitment;

/// The WHIR scheme under test.
type Pcs = WhirProver<EF, F, Radix2DFTSmallBatch<F>, ValMmcs, Challenger, SuffixProver<F, EF>>;

/// A WHIR scheme for one multilinear polynomial, its opening protocol, and an opening point.
fn setup() -> (Pcs, OpeningProtocol, Point<EF>) {
    let folding_factor = FoldingFactor::Constant(FOLDING);
    let (num_rounds, _) = folding_factor
        .compute_number_of_rounds(NUM_VARIABLES)
        .unwrap();
    let round_log_inv_rates = (1..=num_rounds)
        .map(|round| 1 + round * (FOLDING - 1))
        .collect();
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 8,
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        round_log_inv_rates,
    };
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(NUM_VARIABLES, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
    .pad_to_min_num_variables(FOLDING);
    let config = WhirConfig::<EF, F, Challenger>::new(NUM_VARIABLES, params).unwrap();
    let dft = Radix2DFTSmallBatch::<F>::new(1 << config.max_fft_size());
    let perm = default_koalabear_poseidon2_16();
    let mmcs = ValMmcs::new(
        common::Hash::new(perm.clone()),
        common::Compress::new(perm),
        0,
    );
    let point = Point::new(
        (0..NUM_VARIABLES)
            .map(|i| EF::from_basis_coefficients_fn(|j| F::from_usize(7 * i + j + 3)))
            .collect(),
    );
    (Pcs::new(config, dft, mmcs), protocol, point)
}

/// A commitment to a fixed polynomial and its opening at the point, checked natively.
fn commit_and_open(
    pcs: &Pcs,
    protocol: &OpeningProtocol,
    point: &Point<EF>,
) -> (Commitment, PcsProof<F, EF, ValMmcs>) {
    let values = (0..1u32 << NUM_VARIABLES)
        .map(|i| F::from_u32(i * i + 7))
        .collect();
    let table = Table::new(RowMajorMatrix::new(values, 1 << NUM_VARIABLES));
    let witness = SuffixProver::<F, EF>::new_witness(vec![table], FOLDING);
    let challenger = Challenger::new(default_koalabear_poseidon2_16());

    let mut prover = challenger.clone();
    let (commitment, data) = pcs.commit(witness, &mut prover).unwrap();
    let proof = pcs
        .open_at(data, protocol, core::slice::from_ref(point), &mut prover)
        .unwrap();

    let mut verifier = challenger;
    observe_commitment::<F, _, _>(&mut verifier, commitment.clone());
    pcs.verify_at(
        &commitment,
        &proof,
        protocol,
        core::slice::from_ref(point),
        &mut verifier,
    )
    .expect("the opening verifies natively");
    (commitment, proof)
}

/// The circuit verifying an opening, with the evaluations it returns.
///
/// The commitment and the point are constants of the circuit.
fn verify_in_circuit(
    pcs: &Pcs,
    protocol: &OpeningProtocol,
    point: &Point<EF>,
    commitment: &Commitment,
    proof: &PcsProof<F, EF, ValMmcs>,
) -> Result<(Circuit, Vec<EF>), p3_whir::VerifierError> {
    // The transcript binds the commitment before the mirror starts, so it is a constant of the circuit.
    let mut challenger = Challenger::new(default_koalabear_poseidon2_16());
    observe_commitment::<F, _, _>(&mut challenger, commitment.clone());

    let mut arith = Circuit::new(chip());
    let mut mirror = challenger.mirror(&arith);
    let target = pcs.mmcs().load_key_commitment(&mut arith, commitment);
    let point = Point::new(point.iter().map(|&c| Ext::from(c)).collect());
    let evals = pcs.verify_at_in(&mut arith, &target, proof, protocol, &[point], &mut mirror)?;
    drop(mirror);
    let evals = evals[0]
        .current()
        .iter()
        .map(|eval| arith.value_ext(eval))
        .collect();
    Ok((arith, evals))
}

#[test]
fn a_whir_opening_verifies_inside_a_circuit() {
    let (pcs, protocol, point) = setup();
    let (commitment, proof) = commit_and_open(&pcs, &protocol, &point);

    let (arith, evals) = verify_in_circuit(&pcs, &protocol, &point, &commitment, &proof)
        .expect("the opening verifies inside the circuit");
    assert_eq!(evals, proof.evals[0].current());

    let circuit = arith.build();
    for (air, trace) in circuit.airs.iter().zip(&circuit.traces) {
        let name = match air {
            CircuitAir::Value(_) => "value",
            CircuitAir::Public(_) => "public",
            CircuitAir::BaseAlu(_) => "base alu",
            CircuitAir::ExtAlu(_) => "ext alu",
            CircuitAir::Recompose(_) => "recompose",
            CircuitAir::Permute(_) => "permute",
        };
        std::println!("{name}: {} rows", trace.height());
    }
    prove_and_verify(&config(2), &circuit);
}

#[test]
fn a_tampered_whir_opening_is_rejected_inside_the_circuit() {
    let (pcs, protocol, point) = setup();
    let (commitment, mut proof) = commit_and_open(&pcs, &protocol, &point);

    // The claimed evaluation moves off the committed polynomial's value at the point.
    let batch = &proof.evals[0];
    let mut current = batch.current().to_vec();
    current[0] += EF::ONE;
    proof.evals[0] = OpeningBatch::new(current, batch.next().to_vec());

    assert!(verify_in_circuit(&pcs, &protocol, &point, &commitment, &proof).is_err());
}

#[test]
fn uniform_bit_draws_mirror_the_native_challenger() {
    let mut native = Challenger::new(default_koalabear_poseidon2_16());
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let mut mirror = native.clone().mirror(&arith);

    // Widths past 24 bits take two draws of half the bits each.
    for bits in [0, 1, 7, 20, 24, 26, 30] {
        let expected = native.sample_uniform_bits::<true>(bits).unwrap();
        assert_eq!(mirror.sample_uniform_bits::<true>(bits).unwrap(), expected);
        assert_eq!(arith.sampled_index(expected, bits).len(), bits);
    }
    drop(mirror);
    prove_and_verify(&config(2), &arith.build());
}
