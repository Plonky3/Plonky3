pub mod air;
pub mod trace;

const AIR_WIDTH: usize = 2;

#[cfg(test)]
mod tests {
    use crate::air::FibonacciAir;
    use crate::trace::generate_fibonacci_trace;
    use ff::PrimeField;
    use p3_baby_bear::BabyBear;
    use p3_bn254_fr::{Bn254Fr, FFBn254Fr, Poseidon2Bn254};
    use p3_challenger::{
        DuplexChallenger, HashChallenger, MultiField32Challenger, SerializingChallenger32,
    };
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::Field;
    use p3_fri::{create_test_fri_config, FriConfig, TwoAdicFriPcs};
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::{
        CompressionFunctionFromHasher, CryptographicPermutation, MultiField32PaddingFreeSponge,
        PaddingFreeSponge, SerializingHasher32, TruncatedPermutation,
    };
    use p3_uni_stark::{prove, verify, StarkConfig};
    use rand::thread_rng;
    use std::marker::PhantomData;

    use p3_commit::testing::TrivialPcs;
    use zkhash::{
        ark_ff::{BigInteger, PrimeField as _},
        fields::bn256::FpBN256 as ark_FpBN256,
        poseidon2::poseidon2_instance_bn256::RC3,
    };

    // Your choice of Field
    type Val = Bn254Fr;

    // This creates a cubic extension field over Val using a binomial basis. It's used for generating challenges in the proof system.
    // The reason why we want to extend our field for Challenges, is because the original Field size is too small to be brute-forced to solve the challenge.
    type Challenge = Bn254Fr;

    //
    type Perm = Poseidon2Bn254<3>;

    // Your choice of Hash Function
    type MyHash = PaddingFreeSponge<Perm, 3, 2, 1>;

    // Defines a compression function type using ByteHash, with 2 input blocks and 32-byte output.
    type MyCompress = TruncatedPermutation<Perm, 2, 1, 3>;

    // Defines a Merkle tree commitment scheme for field elements with 32 levels.
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 1>;

    // Defines an extension of the Merkle tree commitment scheme for the challenge field.
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

    type Dft = Radix2DitParallel<Val>;

    // Defines the challenger type for generating random challenges.
    type Challenger = HashChallenger<Val, MyHash, 1>;

    // Defines the polynomial commitment scheme type.
    type Pcs = TrivialPcs<Val, Dft>;

    // Defines the overall STARK configuration type.
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    #[test]
    fn it_works() {
        let perm = Perm::new_from_rng(8, 22, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash.clone(), compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft::default();

        // Configures the FRI (Fast Reed-Solomon IOP) protocol parameters.
        let fri_config = FriConfig {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 100,
            proof_of_work_bits: 16,
            mmcs: challenge_mmcs,
        };

        let pcs = TrivialPcs { // TODO
            dft,
            log_n: 3, // TODO
            _phantom: PhantomData,
        };

        //let pcs = Pcs::new(dft, val_mmcs, fri_config);
        let config = MyConfig::new(pcs);

        // Generate the execution trace, based on the inputs defined above.
        let trace = generate_fibonacci_trace::<Bn254Fr>(8);
        println!("trace: {:?}", trace);

        // Create Challenge sequence, in this case, we are using empty vector as seed inputs.
        let mut challenger = Challenger::new(vec![], hash.clone());

        let proof = prove(
            &config,
            &FibonacciAir {
                num_steps: 8,
                final_value: 21,
            },
            &mut challenger,
            trace,
            &vec![],
        );

        // Create the same Challenge sequence as above for verification purpose
        let mut challenger = Challenger::new(vec![], hash.clone());
        // Verify your proof!
        verify(
            &config,
            &FibonacciAir {
                num_steps: 8,
                final_value: 21,
            },
            &mut challenger,
            &proof,
            &vec![],
        )
        .unwrap();
    }
}
