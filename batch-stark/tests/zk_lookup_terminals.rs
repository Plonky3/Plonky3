#[path = "common/config.rs"]
#[allow(dead_code)]
mod config;

use config::{Challenge, MyHidingConfig, Val, make_config_zk};
use p3_air::{Air, AirBuilder, BaseAir, PermutationAirBuilder, WindowAccess};
use p3_batch_stark::config::Commitment;
use p3_batch_stark::{
    BatchShape, BatchVerifierTranscript, ProverData, StarkGenericConfig, StarkInstance,
    prove_batch, verify_batch,
};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::{Count, InteractionBuilder, LogUpGadget, Lookup, LookupProtocol};
use p3_matrix::dense::RowMajorMatrix;

const LOG_HEIGHT: usize = 8;
const HEIGHT: usize = 1 << LOG_HEIGHT;

// Two AIRs sharing the global bus `byte`.
//
//     Sender  columns [x, g]  sends (x) counted by g, with g boolean
//     Table   columns [v, m]  provides (v) counted by -m
#[derive(Clone, Copy, Debug)]
enum ByteBusAir {
    Sender,
    Table,
}

impl<F> BaseAir<F> for ByteBusAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB> Air<AB> for ByteBusAir
where
    AB: AirBuilder + PermutationAirBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let (value, count) = (local[0], local[1]);
        match self {
            Self::Sender => {
                builder.assert_bool(count);
                builder.push_interaction("byte", [value.into()], Count::bounded(count.into(), 1));
            }
            Self::Table => {
                let count: AB::Expr = count.into();
                builder.push_interaction("byte", [value.into()], Count::provided(-count));
            }
        }
    }
}

/// The sender looks up one secret byte on its first row; every other row is padding.
fn sender_trace(secret: u8) -> RowMajorMatrix<Val> {
    let mut values = Val::zero_vec(HEIGHT * 2);
    values[0] = Val::from_u8(secret);
    values[1] = Val::ONE;
    RowMajorMatrix::new(values, 2)
}

/// The table lists every byte once, with multiplicity one on the secret.
fn table_trace(secret: u8) -> RowMajorMatrix<Val> {
    let mut values = Val::zero_vec(HEIGHT * 2);
    for row in 0..HEIGHT {
        values[2 * row] = Val::from_usize(row);
    }
    values[2 * usize::from(secret) + 1] = Val::ONE;
    RowMajorMatrix::new(values, 2)
}

/// Prove one secret, then return every candidate byte whose unmasked LogUp sum
/// equals the sender's published terminal.
///
/// Only the proof, the AIRs, the common data and the configuration are read
/// after proving: everything a verifier holds.
fn candidates_matching_published_terminal(secret: u8) -> Vec<u8> {
    let config = make_config_zk(1);
    let airs = [ByteBusAir::Sender, ByteBusAir::Table];
    let degree_bits = LOG_HEIGHT + config.is_zk();
    let prover_data = ProverData::<MyHidingConfig>::from_airs_and_degrees(
        &config,
        &airs,
        &[degree_bits, degree_bits],
    )
    .unwrap();
    let common = &prover_data.common;

    let sender = sender_trace(secret);
    let table = table_trace(secret);
    let traces = [&sender, &table];
    let public_values: Vec<Vec<Val>> = vec![vec![], vec![]];
    let instances = StarkInstance::new_multiple(&airs, &traces, &public_values);
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();
    verify_batch(&config, &airs, &proof, &public_values, common).unwrap();

    // Replay the public transcript up to the lookup challenges.
    let all_lookups: Vec<&[Lookup<Val>]> = common.lookups.iter().map(|l| &**l).collect();
    let shape = BatchShape {
        trace_widths: vec![2, 2],
        public_value_counts: vec![0, 0],
        preprocessed_widths: vec![0, 0],
        has_preprocessed_commitment: false,
        num_lookup_instances: 2,
        lookup_pow_bits: config.lookup_proof_of_work_bits(),
        has_randomization_commitment: true,
        ood_pow_bits: config.ood_proof_of_work_bits(),
    };
    let mut challenger = config.initialise_challenger();
    let mut transcript = BatchVerifierTranscript::<
        <MyHidingConfig as StarkGenericConfig>::Challenger,
        Val,
        Challenge,
        Commitment<MyHidingConfig>,
    >::new(&mut challenger, shape);
    transcript.instance_bindings(&proof.degree_bits);
    transcript.main_phase(proof.commitments.main.clone(), &public_values);
    transcript.preprocessed_phase(None);
    let gadget = LogUpGadget::new();
    let challenges = transcript
        .lookup_phase(&all_lookups, &gadget, proof.lookup_pow_witness)
        .unwrap();
    transcript.abort();

    let published = proof.lookup_terminals[0].unwrap().0;
    (0..=u8::MAX)
        .filter(|&candidate| {
            let (_, terminal) = gadget.generate_permutation::<Val, Challenge>(
                &sender_trace(candidate),
                &None,
                &[],
                &[],
                all_lookups[0],
                &challenges[0],
            );
            terminal.unwrap().0 == published
        })
        .collect()
}

#[test]
fn hiding_terminals_do_not_reveal_a_looked_up_secret() {
    // Fixture state: one secret byte crosses the bus between two AIRs.
    //
    //     published terminal  T_sender
    //     unmasked LogUp sum  1 / (prefix - candidate)   per candidate byte
    //
    // Invariant: under a hiding PCS no candidate's unmasked sum equals the published terminal.
    // An unmasked terminal matches exactly the secret, so a verifier recovers it by enumeration.
    for secret in [0x00, 0x5a, 0xff] {
        let matches = candidates_matching_published_terminal(secret);
        assert!(
            matches.is_empty(),
            "secret {secret:#04x}: the published terminal matches candidates {matches:?}"
        );
    }
}
