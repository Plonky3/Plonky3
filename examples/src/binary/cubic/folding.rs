//! Folding Boolean commitment for 64-bit values and cubic-extension challenges.

use std::time::Instant;

use p3_binary_field::BinaryField2;
use p3_binary_pcs::{
    BinaryPcsConfig, FoldingBooleanTraceData, FoldingBooleanTracePcs, GroupedCodewordMmcs,
};
use p3_blake3::Blake3;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multi_stark::config::MultiStarkConfig;
use p3_multi_stark::{
    MultiStarkProof, ProverInstance, ProverInstances, ReprBackend, VerifierInstance,
    VerifierInstances, prove_with_backend, security_report, setup, verify,
};
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_symmetric::CryptographicHasher;

use super::{
    BinaryFields, BinaryProofError, BinaryProofOptions, BinaryProofReport, Challenge, Compress,
    CubicAir, Hash, HashFamily, PcsIdentity, Val, ValChallenger,
};

/// One runtime-selected hash keeps the large prover generic over arity only.
#[derive(Clone, Copy, Debug)]
struct FoldingHash(HashFamily);

impl CryptographicHasher<u8, [u8; 32]> for FoldingHash {
    const LANES: usize = if Blake3::LANES > Keccak256Hash::LANES {
        Blake3::LANES
    } else {
        Keccak256Hash::LANES
    };

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        match self.0 {
            HashFamily::Keccak256 => Keccak256Hash.hash_iter(input),
            HashFamily::Blake3 => Blake3.hash_iter(input),
        }
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        match self.0 {
            HashFamily::Keccak256 => Keccak256Hash.hash_iter_slices(input),
            HashFamily::Blake3 => Blake3.hash_iter_slices(input),
        }
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        match self.0 {
            HashFamily::Keccak256 => Keccak256Hash.hash_many(input, out),
            HashFamily::Blake3 => Blake3.hash_many(input, out),
        }
    }
}

type BaseMmcs<const N: usize> = GroupedCodewordMmcs<
    MerkleTreeMmcs<Val, u8, Hash<FoldingHash>, Compress<FoldingHash, N>, N, 32>,
>;
type RoundMmcs<const N: usize> = GroupedCodewordMmcs<
    MerkleTreeMmcs<Challenge, u8, Hash<FoldingHash>, Compress<FoldingHash, N>, N, 32>,
>;
type Pcs<const N: usize> = FoldingBooleanTracePcs<Val, Challenge, BaseMmcs<N>, RoundMmcs<N>>;
type FoldingChallenger = ValChallenger<FoldingHash>;

fn challenger(hash: HashFamily) -> FoldingChallenger {
    FoldingChallenger::from_hasher(
        b"p3-examples-binary-hash-air-v1".to_vec(),
        FoldingHash(hash),
    )
}

struct CubicFoldingStarkConfig<const N: usize> {
    pcs: Pcs<N>,
    leaf_elements: usize,
    stacked_variables: usize,
}

impl<const N: usize> MultiStarkConfig for CubicFoldingStarkConfig<N> {
    type Val = Val;
    type Challenge = Challenge;
    type Challenger = FoldingChallenger;
    type Pcs = Pcs<N>;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn collision_resistance_bits(&self) -> Option<usize> {
        Some(128)
    }

    fn min_num_variables(&self) -> usize {
        1
    }

    fn build_witness(&self, tables: Vec<Table<Val>>) -> Vec<Table<Val>> {
        tables
    }

    fn committed_table<'a>(
        &self,
        prover_data: &'a FoldingBooleanTraceData<Val, Challenge, BaseMmcs<N>>,
        table_index: usize,
    ) -> &'a Table<Val> {
        prover_data.table(table_index)
    }
}

fn config<const N: usize>(
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<CubicFoldingStarkConfig<N>, BinaryProofError> {
    let (arity, _) = plan_stacked_layout(&[shape]);
    let absorbed = BitRingSwitch::<Val, Challenge>::ABSORBED;
    let committed = arity.checked_sub(absorbed).ok_or_else(|| {
        BinaryProofError::CubicFolding(format!(
            "bit witness of {arity} variables cannot absorb {absorbed}"
        ))
    })?;
    let schedule = BinaryPcsConfig::try_new_with_folding::<Val, Challenge>(
        committed,
        options.pcs_params(),
        options.folding.min(committed),
    )?;
    let group = match options.leaf_elements {
        Some(elements) if elements.is_power_of_two() => elements,
        Some(elements) => return Err(BinaryProofError::UnsupportedLeafElements(elements)),
        None => 1 << schedule.log_folding_factor(),
    };
    let hash = FoldingHash(options.hash);
    let base = BaseMmcs::<N>::with_group_size(
        MerkleTreeMmcs::new(Hash::new(hash), Compress::new(hash), 0),
        &schedule,
        group,
    );
    let round = RoundMmcs::<N>::with_group_size(
        MerkleTreeMmcs::new(Hash::new(hash), Compress::new(hash), 0),
        &schedule,
        group,
    );
    let base_height = 1usize << (committed + options.log_inv_rate);
    let leaf_elements = base
        .group_size_at(base_height)
        .expect("the message is nonempty");
    let pcs = Pcs::new(schedule, base, round, arity)
        .map_err(|error| BinaryProofError::CubicFolding(error.to_string()))?;
    Ok(CubicFoldingStarkConfig {
        pcs,
        leaf_elements,
        stacked_variables: arity,
    })
}

pub(super) fn preflight_boolean_air_cubic_folding<A: CubicAir>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<f64, BinaryProofError> {
    match options.merkle_arity {
        2 => preflight_with::<A, 2>(air, shape, options),
        4 => preflight_with::<A, 4>(air, shape, options),
        arity => Err(BinaryProofError::UnsupportedMerkleArity(arity)),
    }
}

fn preflight_with<A: CubicAir, const N: usize>(
    air: &A,
    shape: TableShape,
    options: BinaryProofOptions,
) -> Result<f64, BinaryProofError> {
    let config = config::<N>(shape, options)?;
    let (_, vk) = setup(&config, &[air], &mut challenger(options.hash))
        .map_err(|error| BinaryProofError::CubicFolding(error.to_string()))?;
    let instances = VerifierInstances::new(vec![VerifierInstance::new(
        air,
        &vk,
        shape.num_variables(),
        &[],
    )]);
    let report = security_report(&config, &instances).map_err(BinaryProofError::Security)?;
    report
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    Ok(report.security_bits().expect("assessed security"))
}

pub(super) fn prove_boolean_air_cubic_folding<A: CubicAir>(
    air: &A,
    trace: Table<Val>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError> {
    match options.merkle_arity {
        2 => prove_with::<A, 2>(air, trace, options),
        4 => prove_with::<A, 4>(air, trace, options),
        arity => Err(BinaryProofError::UnsupportedMerkleArity(arity)),
    }
}

fn prove_with<A: CubicAir, const N: usize>(
    air: &A,
    trace: Table<Val>,
    options: BinaryProofOptions,
) -> Result<BinaryProofReport, BinaryProofError> {
    let shape = trace.shape();
    let setup_start = Instant::now();
    let config = config::<N>(shape, options)?;
    let (pk, vk) = setup(&config, &[air], &mut challenger(options.hash))
        .map_err(|error| BinaryProofError::CubicFolding(error.to_string()))?;
    let instances = || {
        VerifierInstances::new(vec![VerifierInstance::new(
            air,
            &vk,
            shape.num_variables(),
            &[],
        )])
    };
    let security = security_report(&config, &instances()).map_err(BinaryProofError::Security)?;
    security
        .require_security(options.security_bits)
        .map_err(BinaryProofError::Security)?;
    let security_bits = security.security_bits().expect("assessed security");
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let prove_start = Instant::now();
    let proof = prove_with_backend::<_, _, ReprBackend<BinaryField2, Challenge, true>>(
        &config,
        ProverInstances::new(vec![ProverInstance::new(air, trace, &pk, &[])]),
        options.sumcheck_pow_bits,
        &mut challenger(options.hash),
    )
    .map_err(|error| BinaryProofError::CubicFolding(error.to_string()))?;
    let prove_seconds = prove_start.elapsed().as_secs_f64();

    let serialize_start = Instant::now();
    let bytes = postcard::to_allocvec(&proof).expect("postcard serialization must not fail");
    let serialize_seconds = serialize_start.elapsed().as_secs_f64();
    let deserialize_start = Instant::now();
    let proof: MultiStarkProof<CubicFoldingStarkConfig<N>> =
        postcard::from_bytes(&bytes).expect("postcard round trip must not fail");
    let deserialize_seconds = deserialize_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    verify(
        &config,
        instances(),
        &proof,
        options.sumcheck_pow_bits,
        &mut challenger(options.hash),
    )
    .map_err(|error| BinaryProofError::CubicFolding(error.to_string()))?;
    let verify_seconds = verify_start.elapsed().as_secs_f64();

    Ok(BinaryProofReport {
        rows: 1 << shape.num_variables(),
        width: shape.width(),
        stacked_variables: config.stacked_variables,
        fields: BinaryFields::Gf64Gf192,
        hash: options.hash,
        leaf_elements: config.leaf_elements,
        requested_leaf_elements: options.leaf_elements,
        proof_bytes: bytes.len(),
        prove_seconds,
        verify_seconds,
        setup_seconds,
        security_bits,
        pcs: PcsIdentity::Folding,
        whir: None,
        serialize_seconds,
        deserialize_seconds,
        threads: p3_maybe_rayon::prelude::current_num_threads(),
        witness_seconds: None,
    })
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField2, Gf2};
    use p3_blake3::Blake3;
    use p3_blake3_air::Blake3BinaryAir;
    use p3_keccak::Keccak256Hash;
    use p3_multi_stark::prove;
    use p3_symmetric::CryptographicHasher;

    use super::*;

    #[test]
    fn sliced_cubic_rounds_match_generic_transcript() {
        let air = Blake3BinaryAir::default();
        let trace = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(4), 2);
        let config = config::<2>(trace.shape(), BinaryProofOptions::default()).unwrap();
        let (pk, _) = setup(&config, &[&air], &mut challenger(HashFamily::Blake3)).unwrap();
        let instances =
            |table| ProverInstances::new(vec![ProverInstance::new(&air, table, &pk, &[])]);
        let generic = prove(
            &config,
            instances(trace.clone()),
            0,
            &mut challenger(HashFamily::Blake3),
        )
        .unwrap();
        let sliced = prove_with_backend::<_, _, ReprBackend<BinaryField2, Challenge, true>>(
            &config,
            instances(trace),
            0,
            &mut challenger(HashFamily::Blake3),
        )
        .unwrap();
        assert_eq!(
            postcard::to_allocvec(&generic).unwrap(),
            postcard::to_allocvec(&sliced).unwrap()
        );
    }

    #[test]
    fn selected_hash_matches_each_existing_hash() {
        let message = [7u8; 128];
        let messages = (0..8u8).flat_map(|byte| [byte; 128]).collect::<Vec<_>>();
        for (choice, expected) in [
            (HashFamily::Keccak256, Keccak256Hash.hash_iter(message)),
            (HashFamily::Blake3, Blake3.hash_iter(message)),
        ] {
            let selected = FoldingHash(choice);
            assert_eq!(selected.hash_iter(message), expected);
            let mut hashes = [[0u8; 32]; 8];
            selected.hash_many(&messages, &mut hashes);
            for (byte, hash) in hashes.into_iter().enumerate() {
                let single = [byte as u8; 128];
                let expected = match choice {
                    HashFamily::Keccak256 => Keccak256Hash.hash_iter(single),
                    HashFamily::Blake3 => Blake3.hash_iter(single),
                };
                assert_eq!(hash, expected);
            }
        }
    }
}
