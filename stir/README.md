
# An implementation of the STIR low-degree test

This crate provides an implementation of the protocol proposed in [STIR: Reed–Solomon Proximity Testing with Fewer Queries](https://eprint.iacr.org/2024/390). It partially follows the [implementation](https://github.com/WizardOfMenlo/stir) by the article's co-author Giacomo Fenzi.

Broadly speaking, STIR allows a prover to convince a verifier that a certain codeword is close to a concrete Reed-Solomon code - that is, it is the evaluation over a certain domain of a polynomial satisfying a known degree bound. Much like FRI, it iteratively folds the original codeword throughout various rounds until it reaches the encoding of a polynomial of a pre-established stopping degree, which is then sent in plain. However, the rate of the codes tested in each round progressively decreases instead of remaining constant, which leads to fewer queries and therefore shorter proofs in exchange for a slightly costlier prover.

## How to use

This crate's usage workflow can be roughly outlined as follows:

 1. Create a set of parameters of type `StirParameters`, which among other elements includes the log2 of the degree bound (plus one) that shoud be proved as well as the target security level in bits. Convenience constructors are provided to facilitate handling of folding factors and rates.

 2. Expand the `StirParameters` into a full `StirConfig` using the latter's `new` constructor. This computes many auxiliary configuration elements as well as the individual round configurations.
 
 3. Encode the polynomial of interest and commit to the codeword using the `commit` method, which produces a `StirWitness` for the prover and an MMCS commitment. If desired,  the commitment can be shared with the verifier at this stage, although this is often unnecessary in non-interactive contexts as the commitment is simply observed by the transcript.

 4. Prove the low-degreeness of the polynomial using the `prove` function, which outputs a `StirProof`. Share the proof with the verifier and, if not done earlier, the commitment as well.

 5. On the verifier end, call `verify` on the received commitment and proof, which will return `Ok(())` if verification was successful and `Err(VerificationError)` otherwise.

STIR can also be composed with other protocols, in which case the commitment from step 3 might have been produced elsewhere.

A full end-to-end example (on a single machine) could look as follows:
```
    use p3_stir::{commit, prove, verify, StirParameters, SecurityAssumption, StirConfig};
    use p3_poly::test_utils::rand_poly;
    use p3_stir::test_utils::{test_bb_challenger, test_bb_mmcs_config, BBExt};

    let log_degree = 15;
    let degree = 1 << log_degree - 1;

    // 1. Define the desired parameters
    let parameters = StirParameters::constant_folding_factor(
        15,
        2,
        3,
        4,
        SecurityAssumption::JohnsonBound,
        100,
        20,
        test_bb_mmcs_config(),
    );

    // 2. Expand into a full configuration
    let config = StirConfig::new::<BBExt>(parameters);

    // 3. Commit to the polynomial
    let polynomial = rand_poly(degree);
    let (witness, commitment) = commit(&config, polynomial);

    // 4. Prove low-degreeness    
    let mut prover_challenger = test_bb_challenger();
    let mut verifier_challenger = prover_challenger.clone();

    let proof = prove(&config, witness, commitment, &mut prover_challenger);

    // 5. Verify the proof
    verify(&config, commitment, proof, &mut verifier_challenger).unwrap();
```

## Tests

Thorough tests have been created of both the low-level, core functionality as well as the end-to-end prover and verifier APIs. The polynomials used for end-to-end tests are only of moderate size (of degree at most 2^15 - 1), unlike the larger ones in the provided benchmark. To execute the tests, run:

```cargo test```

Note that running the tests in release mode is orders of magnitude faster:

```cargo test --release```

## Benchmarks

A benchmark is available which checks the performance of `commit`, `prove` and `verify` on two different fields: a quintic extension of `BabyBear` and a quadratic extension of `Goldilocks`. Run it with:

```cargo bench --bench stir --all-features```

## Features

The crate is `no_std`. It provides the feature `test-utils`, which exposes convenience methods to create MMCS configurations, challengers and STIR parameters for the two fields mentioned above. For an example of how these can be used, see the prover or verifier tests or the file `benches/stir.rs`.

## Implementation notes

We highlight some aspects of the implementation:

 - The Fiat-Shamir transform has been implemented with some care, which includes:
    - Observing the public parameters
    - Domain separation (i. e. preceding each transcript operation with a message identifier)
    - Observing of the size each variable-length list before observing the list itself

 - A custom error class `VerificationError` informs a failed called to `verify` of which step verification went wrong at.

 - Thorough tests are provided, including core mathematical functionality and end-to-end proving and verification. Every variant of `VerificationError` is triggered in them.

 - Shake polynomials are used, an optimisation not present in the article but already in place in the co-author's linked implementation. In summary, instead of having the verifier perform a Lagrange interpolation in order to compute the so-called answer polynomial, the prover supplies this polynomial along with a "shake polynomial" which allows the verifier to convince itself that the answer polynomial indeed interpolates the expected values.
 
 - Smaller verifier optimisations, such as batch inversion of points, are used for an extra 2.5-10% performance gain (depending on the field used and the degree of the polynomial).

 - When defining the `StirParameters`, the user has a choice of three `SecurityAssumption`s. These are explained in more detail in the file `proximity_gaps.rs`. Essentially, the more one is willing to assume in terms of conjectures on the distance of Reed-Solomon codes, the more performant the code is. Unconditional settings (i. e. ones not relying on any open conjectures) are available.

 - The crate is generously documented, both in terms of the external API as well as internals, algorithms and structure fields. Especial care has been put into ensuring the indices and other mathematical objects of the code match the article's notation.
 
 - Proof serialisation and deserialisation capabilities are provided.
