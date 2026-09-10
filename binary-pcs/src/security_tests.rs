//! Compare the real prover/verifier grinding sites with the security model. This records
//! runtime calls, so it does not depend on an unmerged typed-transcript vocabulary.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::BinaryField128 as F;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger,
    GrindingChallenger, ResamplingError,
};
use p3_commit::MultilinearPcs;
use p3_security::binary::BinaryPcsRegime;
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::test_util::{MyChallenger, challenger, mmcs};
use crate::{BinaryPcs, BinaryPcsConfig, BinaryPcsParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    FieldSample,
    Grind(usize),
    Check(usize),
    QuerySample,
}

#[derive(Clone)]
struct RecordingChallenger {
    inner: MyChallenger,
    events: Vec<Event>,
}

impl RecordingChallenger {
    fn new() -> Self {
        Self {
            inner: challenger(),
            events: Vec::new(),
        }
    }
}

impl<T> CanObserve<T> for RecordingChallenger
where
    MyChallenger: CanObserve<T>,
{
    fn observe(&mut self, value: T) {
        self.inner.observe(value);
    }
}
impl CanSample<F> for RecordingChallenger {
    fn sample(&mut self) -> F {
        self.events.push(Event::FieldSample);
        self.inner.sample()
    }
}
impl CanSampleBits<usize> for RecordingChallenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.inner.sample_bits(bits)
    }
}
impl CanSampleUniformBits<F> for RecordingChallenger {
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        self.events.push(Event::QuerySample);
        self.inner.sample_uniform_bits::<RESAMPLE>(bits)
    }
}
impl FieldChallenger<F> for RecordingChallenger {}
impl GrindingChallenger for RecordingChallenger {
    type Witness = F;
    fn grind(&mut self, bits: usize) -> F {
        self.events.push(Event::Grind(bits));
        self.inner.grind(bits)
    }
    fn check_witness(&mut self, bits: usize, witness: F) -> bool {
        self.events.push(Event::Check(bits));
        self.inner.check_witness(bits, witness)
    }
}

fn matches_model(
    events: &[Event],
    regime: BinaryPcsRegime,
    field_samples: usize,
    prover: bool,
) -> bool {
    let Some((fields, rest)) = events.split_at_checked(field_samples) else {
        return false;
    };
    let expected = if prover {
        Event::Grind(regime.query_pow_bits())
    } else {
        Event::Check(regime.query_pow_bits())
    };
    fields.iter().all(|&e| e == Event::FieldSample)
        && rest.first() == Some(&expected)
        && rest.len() > 1
        && rest[1..].iter().all(|&e| e == Event::QuerySample)
}

#[test]
fn actual_grinding_matches_the_model_and_follows_alpha_and_every_fold() {
    for pow_bits in [0, 4] {
        for folding in [1, 3] {
            let config = BinaryPcsConfig::try_new(
                8,
                BinaryPcsParams {
                    log_inv_rate: 2,
                    pow_bits,
                    security_level: 100,
                },
            )
            .unwrap()
            .try_with_folding(folding)
            .unwrap();
            let pcs = BinaryPcs::new(config, mmcs());
            let mut rng = SmallRng::seed_from_u64(934);
            let witness = SuffixProver::<F, F>::new_witness(vec![Table::rand(&mut rng, 1, 8)], 0);
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(8, 1),
                vec![OpeningBatch::new(vec![0], vec![0])],
            )]);
            let mut pc = RecordingChallenger::new();
            let (root, data) = pcs.commit(witness, &mut pc);
            let proof = pcs.try_open(data, &protocol, &mut pc).unwrap();
            let mut vc = RecordingChallenger::new();
            pcs.verify(&root, &proof, &mut vc, protocol).unwrap();
            // One opening-point seed, alpha, then eight independent fold challenges.
            assert!(
                matches_model(&pc.events, config.security_regime(), 10, true),
                "{:?}",
                pc.events
            );
            assert!(
                matches_model(&vc.events, config.security_regime(), 10, false),
                "{:?}",
                vc.events
            );

            let overcredited =
                BinaryPcsRegime::new(8, 2, folding, config.num_queries(), pow_bits + 1).unwrap();
            assert!(!matches_model(&pc.events, overcredited, 10, true));
            assert!(!matches_model(&vc.events, overcredited, 10, false));
            let mut early = pc.events.clone();
            early.swap(1, 10); // Move the grind before alpha without changing its difficulty.
            assert!(!matches_model(&early, config.security_regime(), 10, true));
        }
    }
}
