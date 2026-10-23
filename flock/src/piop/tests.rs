use alloc::vec::Vec;

use p3_binary_pcs::whir::preset::keccak64::{self, Commitment, OpeningProof};
use p3_binary_pcs::whir::{CommitmentParams, RegionLayout, WeightedOpening};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

use super::*;
use crate::tests::{random_circuit, random_rows};
use crate::zerocheck::MIN_LOG_N;
use crate::{Circuit, Port};

/// A fresh sponge.
fn sponge() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// `sum_v low_weights[v] * t(high, v)` over every committed word.
fn reference(words: &[u64], opening: &WeightedOpening<Challenge>) -> Challenge {
    let eq = Poly::new_from_point(opening.high.as_slice(), Challenge::ONE);
    words
        .iter()
        .zip(eq.as_slice())
        .map(|(&word, &e)| {
            let bits: Challenge = (0..64)
                .filter(|v| word >> v & 1 == 1)
                .map(|v| opening.low_weights[v])
                .sum();
            e * bits
        })
        .sum()
}

/// A Flock batch in region 1 of a shared commitment, beside a dummy claim on region 0.
struct Shared {
    circuit: Circuit,
    log_instances: usize,
    public: PublicPorts,
    layout: RegionLayout,
    pcs: WitnessCommitment,
    root: Commitment,
    dummy: WeightedClaim<Challenge>,
    piop: PiopProof,
    split: bool,
    opening: OpeningProof,
}

impl Shared {
    /// Commit a batch's witness at region 1 and random words elsewhere, prove, and open both claims at once.
    ///
    /// Region 0 has `dummy_log` variables, the batch's ports are public.
    fn new(seed: u64, dummy_log: impl Fn(usize) -> usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let circuit = random_circuit(&mut rng, 6000);
        let log_instances = MIN_LOG_N.saturating_sub(circuit.k_log()).max(3);
        let (rows, padding) = random_rows(&mut rng, (1 << log_instances) - 5);
        let mut witness =
            Witness::generate(&circuit, log_instances, &rows, &padding, |row, out| {
                out.copy_from_slice(row);
            });
        let batch = Batch::new(&circuit, log_instances).unwrap();
        let instance_words = (1 << circuit.k_log()) / 64;
        let public = PublicPorts::new(&batch, &[Port::Input(0), Port::Output(0)], |i, out| {
            let len = out.len();
            out.copy_from_slice(&witness.z()[i * instance_words..][..len]);
        });

        let layout = RegionLayout::allocate(&[dummy_log(batch.log_n()), batch.log_n()]);
        let [region0, region1] = [layout.regions()[0], layout.regions()[1]];
        let pcs =
            keccak64::witness_commitment(layout.num_variables, &CommitmentParams::default(), 2)
                .unwrap();
        let mut words: Vec<u64> = (0..1 << (layout.num_variables - 6))
            .map(|_| rng.next_u64())
            .collect();
        region1.words_mut(&mut words).copy_from_slice(witness.z());
        let _ = witness.take_z();

        // The dummy claim: a multilinear evaluation on region 0, at a point of its own.
        let point = Point::new((0..region0.log_bits).map(|_| rng.random()).collect());
        let opening =
            WeightedOpening::at_point(&point, 6).within(&region0.selector(layout.num_variables));
        let dummy = WeightedClaim {
            value: reference(&words, &opening),
            opening,
        };

        // Statement, commitment, PIOP, then one opening of both claims.
        let mut ch = sponge();
        public.bind(&batch, &mut ch).unwrap();
        let (root, data) = pcs.commit(words, &mut ch).unwrap();
        let placement = Placement {
            commitment: &pcs,
            region: region1,
        };
        let (piop, claim) =
            prove(0, &batch, &public, &witness, &placement, &data, &mut ch).unwrap();
        let split = claim.split.is_some();
        let opening = pcs
            .open(
                data,
                &[dummy.clone(), claim.claim],
                &[None, claim.split.as_ref()],
                &mut ch,
            )
            .unwrap();
        Self {
            circuit,
            log_instances,
            public,
            layout,
            pcs,
            root,
            dummy,
            piop,
            split,
            opening,
        }
    }

    /// Replay the batch at `region`, then check one opening of the dummy claim and the batch's.
    fn verify(
        &self,
        region: Region,
        dummy: &WeightedClaim<Challenge>,
        piop: &PiopProof,
    ) -> Result<(), VerificationError> {
        let batch = Batch::new(&self.circuit, self.log_instances).unwrap();
        let mut ch = sponge();
        self.public.bind(&batch, &mut ch)?;
        self.pcs.observe(&self.root, &mut ch);
        let placement = Placement {
            commitment: &self.pcs,
            region,
        };
        let claim = verify(0, &batch, &self.public, piop, &placement, &mut ch)?;
        self.pcs
            .verify(&self.root, &[dummy.clone(), claim], &self.opening, &mut ch)?;
        Ok(())
    }

    /// The batch's region.
    fn region(&self) -> Region {
        self.layout.regions()[1]
    }
}

/// Region 0 as wide as the batch, half as wide, and twice as wide: the batch lands at bit `2^n`, `0` and `2^(n+1)`.
const LAYOUTS: [fn(usize) -> usize; 3] = [|n| n, |n| n - 1, |n| n + 1];

#[test]
fn a_batch_in_a_region_verifies_beside_a_dummy_claim() {
    let mut split = false;
    for (seed, layout) in LAYOUTS.into_iter().enumerate() {
        let r = Shared::new(seed as u64, layout);
        r.verify(r.region(), &r.dummy, &r.piop)
            .unwrap_or_else(|e| panic!("layout {seed}: {e:?}"));
        split |= r.split;
    }
    // The folded witness splits the batch's claim, so the deferred switch skips its sweep.
    assert!(split);
}

#[test]
fn a_tampered_batch_or_dummy_claim_is_rejected() {
    let r = Shared::new(0x7A3, LAYOUTS[0]);
    r.verify(r.region(), &r.dummy, &r.piop).unwrap();

    // Mutation: one lincheck slice.
    let mut piop = r.piop.clone();
    piop.lincheck.slices[0] += Challenge::ONE;
    assert!(r.verify(r.region(), &r.dummy, &piop).is_err());

    // Mutation: one zerocheck evaluation.
    let mut piop = r.piop.clone();
    piop.zerocheck.evals[2] += Challenge::ONE;
    assert!(r.verify(r.region(), &r.dummy, &piop).is_err());

    // Mutation: the dummy claim's value.
    let mut dummy = r.dummy.clone();
    dummy.value += Challenge::ONE;
    assert!(matches!(
        r.verify(r.region(), &dummy, &r.piop),
        Err(VerificationError::Opening(_))
    ));
}

#[test]
fn a_batch_replayed_at_another_region_is_rejected() {
    // Mutation: the batch checked at region 0, of its size, so the PIOP replays and the opening refuses.
    let r = Shared::new(0x5E1, LAYOUTS[0]);
    let other = r.layout.regions()[0];
    assert_eq!(other.log_bits, r.region().log_bits);
    assert!(matches!(
        r.verify(other, &r.dummy, &r.piop),
        Err(VerificationError::Opening(_))
    ));

    // Mutation: a region of another size, misaligned, or outside the commitment, before the transcript moves.
    let region = r.region();
    let total = r.layout.num_variables;
    for bad in [
        Region {
            log_bits: region.log_bits + 1,
            offset_bits: 0,
        },
        Region {
            log_bits: region.log_bits,
            offset_bits: region.offset_bits + 64,
        },
        Region {
            log_bits: region.log_bits,
            offset_bits: 1 << total,
        },
    ] {
        assert!(matches!(
            r.verify(bad, &r.dummy, &r.piop),
            Err(VerificationError::Region(_))
        ));
    }
}

#[test]
fn the_prover_refuses_a_region_or_words_off_the_batch() {
    let mut rng = SmallRng::seed_from_u64(0xD07);
    let circuit = random_circuit(&mut rng, 1500);
    let log_instances = MIN_LOG_N.saturating_sub(circuit.k_log()).max(3);
    let (rows, padding) = random_rows(&mut rng, 1 << log_instances);
    let witness = Witness::generate(&circuit, log_instances, &rows, &padding, |row, out| {
        out.copy_from_slice(row);
    });
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let layout = RegionLayout::allocate(&[batch.log_n(), batch.log_n()]);
    let pcs = keccak64::witness_commitment(layout.num_variables, &CommitmentParams::default(), 2)
        .unwrap();
    let mut words = Vec::from_iter(0..1u64 << (layout.num_variables - 6));
    layout.regions()[1]
        .words_mut(&mut words)
        .copy_from_slice(witness.z());
    let mut ch = sponge();
    let (_, data) = pcs.commit(words, &mut ch).unwrap();

    // Public words read from the batch's witness.
    let instance_words = (1 << circuit.k_log()) / 64;
    let public = PublicPorts::new(&batch, &[Port::Input(0)], |i, out| {
        let len = out.len();
        out.copy_from_slice(&witness.z()[i * instance_words..][..len]);
    });

    // Mutation: the region of the commitment's other words, which miss the public words.
    let wrong = Placement {
        commitment: &pcs,
        region: layout.regions()[0],
    };
    assert!(matches!(
        prove(0, &batch, &public, &witness, &wrong, &data, &mut ch),
        Err(ProvingError::Ports(_))
    ));

    // Mutation: the whole commitment, wider than the batch.
    assert!(matches!(
        prove(
            0,
            &batch,
            &public,
            &witness,
            &Placement::whole(&pcs),
            &data,
            &mut ch
        ),
        Err(ProvingError::Region(_))
    ));
}
