use alloc::collections::VecDeque;
use alloc::vec;

use p3_baby_bear::BabyBear;
use p3_challenger::{CanSampleBits, ResamplingError};
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use super::*;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

#[derive(Clone)]
struct Scripted {
    samples: VecDeque<F>,
    observed: Vec<F>,
}

impl CanObserve<F> for Scripted {
    fn observe(&mut self, value: F) {
        self.observed.push(value);
    }
}
impl CanSample<F> for Scripted {
    fn sample(&mut self) -> F {
        self.samples.pop_front().expect("script exhausted")
    }
}
impl CanSampleBits<usize> for Scripted {
    fn sample_bits(&mut self, _: usize) -> usize {
        panic!("unexpected bit draw")
    }
}
impl CanSampleUniformBits<F> for Scripted {
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        _: usize,
    ) -> Result<usize, ResamplingError> {
        panic!("unexpected uniform-bit draw")
    }
}
impl GrindingChallenger for Scripted {
    type Witness = F;
    fn grind(&mut self, _: usize) -> F {
        panic!("unexpected grind")
    }
}

fn ood_separator() -> DomainSeparator<Alphabet<F>> {
    let mut steps = Vec::new();
    for _ in 0..2 {
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            OOD_POINT,
            Length::Fixed(1),
        ));
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Message,
            OOD_ANSWER,
            Length::Scalar,
        ));
    }
    DomainSeparator::new(VERSION, NAME, InteractionPattern::new(steps).unwrap())
}

#[test]
fn zero_and_duplicate_ood_candidates_are_resampled_symmetrically() {
    // Reject zero, then a duplicate and zero. Leave the sentinel unread.
    let scripted = Scripted {
        samples: [0, 7, 7, 0, 9, 13]
            .into_iter()
            .flat_map(|n| EF::from_u32(n).as_basis_coefficients_slice().to_vec())
            .collect(),
        observed: vec![],
    };
    let mut prover = scripted.clone();
    let mut verifier = scripted;
    let separator = ood_separator();
    let mut p = ZkWhirProverTranscript::<_, F, EF>::from_separator(&mut prover, &separator);
    let mut v = ZkWhirVerifierTranscript::<_, F, EF>::from_separator(&mut verifier, &separator);
    let mut kept = Vec::new();
    for expected in [EF::from_u32(7), EF::from_u32(9)] {
        assert_eq!(p.ood_point(&kept), expected);
        assert_eq!(v.ood_point(&kept), expected);
        p.observe(OOD_ANSWER, EF::ONE);
        v.observe(OOD_ANSWER, EF::ONE);
        kept.push(expected);
    }
    p.finish();
    v.finish();
    assert_eq!(prover.samples, verifier.samples);
    assert_eq!(prover.samples.len(), 4);
    assert_eq!(prover.sample(), F::from_u32(13));
    assert_eq!(prover.observed, verifier.observed);
}

#[test]
fn incomplete_players_reject_finish_and_verifier_can_abort() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut challenger = Scripted {
        samples: VecDeque::new(),
        observed: vec![],
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            ZkWhirProverTranscript::<_, F, EF>::from_separator(&mut challenger, &ood_separator())
                .finish();
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            ZkWhirVerifierTranscript::<_, F, EF>::from_separator(&mut challenger, &ood_separator())
                .finish();
        }))
        .is_err()
    );
    let mut verifier =
        ZkWhirVerifierTranscript::<_, F, EF>::from_separator(&mut challenger, &ood_separator());
    verifier.abort();
}

#[test]
fn reordered_ood_answer_is_rejected_before_absorption() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut challenger = Scripted {
        samples: VecDeque::new(),
        observed: vec![],
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut prover = ZkWhirProverTranscript::<_, F, EF>::from_separator(
                &mut challenger,
                &ood_separator(),
            );
            prover.observe(OOD_ANSWER, EF::ONE);
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut verifier = ZkWhirVerifierTranscript::<_, F, EF>::from_separator(
                &mut challenger,
                &ood_separator(),
            );
            verifier.observe(OOD_ANSWER, EF::ONE);
        }))
        .is_err()
    );
}

#[test]
fn unfinished_players_cannot_be_silently_dropped() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut challenger = Scripted {
        samples: VecDeque::new(),
        observed: vec![],
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            drop(ZkWhirProverTranscript::<_, F, EF>::from_separator(
                &mut challenger,
                &ood_separator(),
            ));
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            drop(ZkWhirVerifierTranscript::<_, F, EF>::from_separator(
                &mut challenger,
                &ood_separator(),
            ));
        }))
        .is_err()
    );
}
