//! The Boolean commitment over every Merkle layout the commitment scheme offers.
//!
//! ```text
//!     cap height   how many layers a path stops short of the root
//!     grouping     how many adjacent symbols share one leaf
//! ```
//!
//! Both are the caller's choice of commitment scheme, taken as a type parameter.
//! This file is what pins that choice to be free.

use p3_binary_field::{BinaryChallenger, BinaryField128, Gf2, PackedGf2x64};
use p3_binary_pcs::{
    BinaryPcsConfig, BinaryPcsParams, BooleanMultilinearPcs, BooleanPcs, GroupedCodewordMmcs,
};
use p3_challenger::HashChallenger;
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type Grouped = GroupedCodewordMmcs<MyMmcs>;
type MyChallenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Bits the round-trip fixture commits, in log bits of one Boolean column.
const LOG_BITS: usize = 13;

/// Bits the size fixture commits, tall enough that the sampled queries stay sparse.
///
/// At the shorter height every coset of the base word is queried.
/// The pruned frontier then covers the whole tree, and no layout can shorten anything.
const LOG_TALL_BITS: usize = 17;

/// Coordinates one element of the widest level absorbs.
const ABSORBED: usize = 7;

const fn mmcs(cap_height: usize) -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        cap_height,
    )
}

const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

const fn params() -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    }
}

fn config(log_bits: usize, log_folding_factor: usize) -> BinaryPcsConfig {
    BinaryPcsConfig::try_new_with_folding::<EF, EF>(
        log_bits - ABSORBED,
        params(),
        log_folding_factor,
    )
    .unwrap()
}

/// A random bit-sliced witness covering the committed hypercube.
fn witness(seed: u64, log_bits: usize) -> Vec<PackedGf2x64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..1 << (log_bits - 6))
        .map(|_| PackedGf2x64::new(rng.random::<u64>()))
        .collect()
}

/// The witness as a multilinear over every variable, one element per bit.
fn embedded(bits: &[PackedGf2x64]) -> Poly<EF> {
    Poly::new(
        bits.iter()
            .flat_map(|block| {
                (0..PackedGf2x64::WIDTH).map(move |lane| {
                    if block.get(lane) == Gf2::ONE {
                        EF::ONE
                    } else {
                        EF::ZERO
                    }
                })
            })
            .collect::<Vec<EF>>(),
    )
}

/// Two opening points the fixture reuses, so every layout answers for the same claims.
fn points(seed: u64, log_bits: usize) -> Vec<Point<EF>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..2)
        .map(|_| Point::<EF>::rand(&mut rng, log_bits))
        .collect()
}

#[test]
fn a_boolean_opening_round_trips_at_every_cap_height() {
    // Invariant: a cap moves the commitment down a layer and moves no value.
    //
    //     cap 0  ->  the commitment is the root
    //     cap 4  ->  the commitment is the layer four levels below it
    let bits = witness(0xCA90, LOG_BITS);
    let points = points(0xCA91, LOG_BITS);
    let reference = embedded(&bits);

    for cap_height in [0usize, 2, 4] {
        let config = config(LOG_BITS, 1);
        let pcs = BooleanPcs::new(config, mmcs(cap_height), mmcs(cap_height), LOG_BITS).unwrap();

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        // The values answer for the bits, whatever the Merkle layout is.
        for (point, &value) in points.iter().zip(&values) {
            assert_eq!(value, reference.eval_base(point), "cap {cap_height}");
        }

        let mut verifier_chal = challenger();
        pcs.observe_commitment(&commitment, &mut verifier_chal);
        pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
            .unwrap_or_else(|error| panic!("cap {cap_height}: {error:?}"));
    }
}

#[test]
fn a_boolean_opening_round_trips_over_grouped_leaves() {
    // Invariant: grouping adjacent symbols into one leaf moves no value either.
    //
    //     one symbol per leaf  ->  one path per symbol of a queried coset
    //     one coset per leaf   ->  one path per coset
    //
    // The fold reads a whole coset per query.
    // A coset-sized leaf is therefore the layout under which one query reads one run.
    let bits = witness(0x6700, LOG_BITS);
    let points = points(0x6701, LOG_BITS);
    let reference = embedded(&bits);

    for log_folding_factor in [1usize, 2, 3] {
        let config = config(LOG_BITS, log_folding_factor);
        let pcs = BooleanPcs::new(
            config,
            Grouped::for_folding(mmcs(0), &config),
            Grouped::for_folding(mmcs(0), &config),
            LOG_BITS,
        )
        .unwrap();

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        for (point, &value) in points.iter().zip(&values) {
            assert_eq!(
                value,
                reference.eval_base(point),
                "arity {log_folding_factor}"
            );
        }

        let mut verifier_chal = challenger();
        pcs.observe_commitment(&commitment, &mut verifier_chal);
        pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
            .unwrap_or_else(|error| panic!("arity {log_folding_factor}: {error:?}"));
    }
}

/// One two-point opening over the tall fixture, returning the serialized proof size.
///
/// The opened values are checked against the bit witness on the way.
/// A layout that moved a value would fail here rather than only change a number.
fn tall_opening<M>(config: BinaryPcsConfig, base: M, rounds: M, label: &str) -> usize
where
    M: Mmcs<EF> + Clone,
    M::Error: core::fmt::Debug,
    MyChallenger: p3_challenger::CanObserve<M::Commitment>,
{
    let bits = witness(0x5121, LOG_TALL_BITS);
    let points = points(0x5122, LOG_TALL_BITS);
    let reference = embedded(&bits);

    let pcs = BooleanPcs::new(config, base, rounds, LOG_TALL_BITS).unwrap();
    let mut chal = challenger();
    let (commitment, data) = pcs.commit_bits(&bits, &mut chal).unwrap();
    let (values, proof) = pcs.open_at_points(data, &points, &mut chal).unwrap();

    for (point, &value) in points.iter().zip(&values) {
        assert_eq!(value, reference.eval_base(point), "{label}");
    }
    let mut verifier_chal = challenger();
    pcs.observe_commitment(&commitment, &mut verifier_chal);
    pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
        .unwrap_or_else(|error| panic!("{label}: {error:?}"));

    postcard::to_allocvec(&proof).unwrap().len()
}

#[test]
fn every_merkle_layout_opens_the_same_committed_bits() {
    // Invariant: the Merkle layout is the caller's choice and changes no value.
    //
    //     - arity   how many folds share one committed word
    //     - cap     how many layers a path stops short of the root
    //     - group   whether a whole coset shares one leaf
    //
    // Every cell of this grid answers for the same bits at the same two points.
    //
    // The sizes are compared against the plain, uncapped cell rather than ordered.
    // The multi-opening already prunes to the minimal frontier.
    //
    //     - a cap mostly widens the committed words it covers
    //     - a grouped leaf mostly replaces paths the frontier had already merged
    //
    // Neither moves this proof by much, in either direction.
    for log_folding_factor in [1usize, 3] {
        let config = config(LOG_TALL_BITS, log_folding_factor);
        let baseline = tall_opening(config, mmcs(0), mmcs(0), "baseline");

        for cap_height in [0usize, 4] {
            let label = format!("arity {log_folding_factor}, cap {cap_height}");
            let plain = tall_opening(config, mmcs(cap_height), mmcs(cap_height), &label);
            let grouped = tall_opening(
                config,
                Grouped::for_folding(mmcs(cap_height), &config),
                Grouped::for_folding(mmcs(cap_height), &config),
                &label,
            );

            // A fifth of the baseline is far wider than any measured spread.
            let bound = baseline / 5;
            assert!(
                plain.abs_diff(baseline) < bound,
                "{label}: {plain} vs {baseline}"
            );
            assert!(
                grouped.abs_diff(baseline) < bound,
                "{label} grouped: {grouped} vs {baseline}"
            );
        }
    }
}
