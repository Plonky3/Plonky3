//! Port check tests: honest statements verify, and every changed, swapped or unbound public word is rejected.

use alloc::vec::Vec;

use p3_keccak::Keccak256Hash;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::{PortsError, PublicPorts};
use crate::lincheck::LincheckError;
use crate::opening::WitnessPcs;
use crate::proof::FlockProof;
use crate::tests::{random_circuit, random_rows};
use crate::zerocheck::MIN_LOG_N;
use crate::{
    Batch, Circuit, FlockChallenger, FlockConfig, Port, ProvingError, VerificationError, Witness,
    lincheck, prove, verify, zerocheck,
};

/// A fresh sponge.
fn sponge() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A random circuit with its witness over enough instances for the zerocheck.
fn fixture(seed: u64) -> (Circuit, Witness, usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let c = random_circuit(&mut rng, 1500);
    let log_n = MIN_LOG_N.saturating_sub(c.k_log()).max(4);
    let (rows, padding) = random_rows(&mut rng, (1 << log_n) - 3);
    let w = Witness::generate(&c, log_n, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    (c, w, log_n)
}

/// The given ports of every instance, read from the witness.
fn from_witness(batch: &Batch<'_>, w: &Witness, ports: &[Port]) -> PublicPorts {
    let words = 1 << (batch.circuit().k_log() - 6);
    PublicPorts::new(batch, ports, |i, out| {
        let len = out.len();
        out.copy_from_slice(&w.z()[i * words..][..len]);
    })
}

/// Every port of the fixture's circuit.
const ALL: [Port; 5] = [
    Port::Input(0),
    Port::Input(1),
    Port::Input(2),
    Port::Output(0),
    Port::Output(1),
];

#[test]
fn honest_statements_verify_for_every_choice_of_ports() {
    let (c, w, log_n) = fixture(0x9047);
    let batch = Batch::new(&c, log_n).unwrap();
    let config = FlockConfig::default();
    for ports in [
        &ALL[..],
        &[Port::Input(1)],
        &[Port::Output(0)],
        &[Port::Output(1), Port::Input(0)],
        &[],
    ] {
        let public = from_witness(&batch, &w, ports);
        let proof = prove(&config, &batch, &public, &w, &mut sponge()).unwrap();
        verify(&config, &batch, &public, &proof, &mut sponge()).unwrap();
    }
}

#[test]
fn a_changed_swapped_or_unbound_public_word_is_rejected() {
    let (c, w, log_n) = fixture(0xBAD9);
    let batch = Batch::new(&c, log_n).unwrap();
    let config = FlockConfig::default();
    let public = from_witness(&batch, &w, &ALL);
    let proof = prove(&config, &batch, &public, &w, &mut sponge()).unwrap();
    let rejects = |public: &PublicPorts, proof: &FlockProof| {
        verify(&config, &batch, public, proof, &mut sponge()).is_err()
    };

    // Mutation: one byte of one output word of the last instance.
    let mut changed = public.clone();
    let last = (changed.values.len() / changed.stride - 1) * changed.stride;
    let output = changed
        .words
        .iter()
        .position(|&w| w >= c.n_input_words())
        .unwrap();
    changed.values[last + output] ^= 0x80 << 8;
    assert!(rejects(&changed, &proof));

    // Mutation: two instances swap their public words.
    let mut swapped = public.clone();
    let (first, second) = swapped.values.split_at_mut(swapped.stride);
    first.swap_with_slice(&mut second[..swapped.stride]);
    assert_ne!(swapped, public);
    assert!(rejects(&swapped, &proof));

    // Mutation: a statement with one more public port than the proof bound.
    let fewer = from_witness(&batch, &w, &ALL[..4]);
    let narrow = prove(&config, &batch, &fewer, &w, &mut sponge()).unwrap();
    assert!(rejects(&public, &narrow));

    // Mutation: a bare proof claimed for public words, and the other way round.
    let bare = prove(&config, &batch, &PublicPorts::default(), &w, &mut sponge()).unwrap();
    assert!(rejects(&public, &bare));
    assert!(rejects(&PublicPorts::default(), &proof));

    // Mutation: public words made for another batch size are refused before anything is sent.
    let wider = Batch::new(&c, log_n + 1).unwrap();
    assert!(matches!(
        verify(&config, &wider, &public, &proof, &mut sponge()),
        Err(VerificationError::Ports(PortsError::InstanceCount { .. }))
    ));
}

#[test]
fn an_honest_prover_refuses_a_statement_its_witness_breaks() {
    let (c, w, log_n) = fixture(0x2EF);
    let batch = Batch::new(&c, log_n).unwrap();
    let mut public = from_witness(&batch, &w, &ALL);
    public.values[3 * public.stride] ^= 1;
    assert!(matches!(
        prove(&FlockConfig::default(), &batch, &public, &w, &mut sponge()),
        Err(ProvingError::Ports(PortsError::Mismatch { instance: 3 }))
    ));
}

#[test]
fn a_prover_whose_witness_breaks_the_statement_is_rejected() {
    // A prover skipping its own consistency check: the witness satisfies the circuit, the statement lies about one word.
    let (c, w, log_n) = fixture(0xF0E);
    let batch = Batch::new(&c, log_n).unwrap();
    let config = FlockConfig::default();
    let mut public = from_witness(&batch, &w, &[Port::Input(0), Port::Output(0)]);
    public.values[5 * public.stride + 1] ^= 1 << 17;

    let mut ch = sponge();
    public.bind(&batch, &mut ch).unwrap();
    let pcs = WitnessPcs::new(batch.log_n(), &config.pcs).unwrap();
    let (commitment, data) = pcs.commit(&w, &mut ch).unwrap();
    let (zerocheck, claims) = zerocheck::prove(&batch, &w, config.pow_bits, &mut ch);
    let (lincheck, claim) = lincheck::prove(&batch, &w, &claims, &public, &mut ch);
    let opening = pcs.open(data, &[claim], &[], &mut ch).unwrap();
    let forged = FlockProof {
        commitment,
        zerocheck,
        lincheck,
        opening,
    };
    assert!(matches!(
        verify(&config, &batch, &public, &forged, &mut sponge()),
        Err(VerificationError::Lincheck(LincheckError::FinalCheck))
    ));
}

#[test]
fn ports_lie_at_the_start_of_an_instance() {
    // Invariant: inputs from word zero in order, then outputs, then the constant.
    let (c, ..) = fixture(0x1);
    let mut end = 0;
    for port in ALL {
        let words = c.port(port).unwrap();
        assert_eq!(words.start, end, "{port:?}");
        end = words.end;
    }
    assert_eq!(64 * end, c.const_pos());
    assert_eq!(c.port(Port::Input(3)), None);
}
